from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        attention_eval: bool = False,
        attention_eval_mode: str = "both",
        attention_eval_ratios: tuple[float, ...] = (0.01, 0.05, 0.10, 0.50),
        tts_mode: str = "none",
        tts_num_candidates: int = 4,
        tts_selection_ratio: float = 0.10,
        tts_branch_ratio: float = 0.40,
        tts_branch_noise_scale: float = 0.10,
        tts_score_mode: str = "mae",
        flow_mg_mask: str = "language",
        flow_mg_steps: tuple[int, ...] = (4, 7, 9),
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._attention_eval = attention_eval
        self._attention_eval_mode = attention_eval_mode
        self._attention_eval_ratios = attention_eval_ratios
        self._tts_mode = tts_mode
        self._tts_num_candidates = tts_num_candidates
        self._tts_selection_ratio = tts_selection_ratio
        self._tts_branch_ratio = tts_branch_ratio
        self._tts_branch_noise_scale = tts_branch_noise_scale
        self._tts_score_mode = tts_score_mode
        self._flow_mg_mask = flow_mg_mask
        self._flow_mg_steps = flow_mg_steps

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._sample_actions_with_attention_metrics = (
                nnx_utils.module_jit(
                    model.sample_actions_with_attention_metrics,
                    static_argnames=(
                        "metric_mode",
                        "metric_ratios",
                        "tts_mode",
                        "num_candidates",
                        "selection_ratio",
                        "branch_ratio",
                        "branch_noise_scale",
                        "tts_score_mode",
                        "flow_mg_mask",
                        "flow_mg_steps",
                    ),
                )
                if attention_eval
                else None
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if self._attention_eval:
            if self._is_pytorch_model:
                raise NotImplementedError(
                    "Attention metric evaluation is currently implemented only for JAX pi0 models."
                )
            sampled = self._sample_actions_with_attention_metrics(
                sample_rng_or_pytorch_device,
                observation,
                metric_mode=self._attention_eval_mode,
                metric_ratios=self._attention_eval_ratios,
                tts_mode=self._tts_mode,
                num_candidates=self._tts_num_candidates,
                selection_ratio=self._tts_selection_ratio,
                branch_ratio=self._tts_branch_ratio,
                branch_noise_scale=self._tts_branch_noise_scale,
                tts_score_mode=self._tts_score_mode,
                flow_mg_mask=self._flow_mg_mask,
                flow_mg_steps=self._flow_mg_steps,
                **sample_kwargs,
            )
            outputs = {
                "state": inputs["state"],
                "actions": sampled["actions"],
                "attention_metrics": sampled["attention_metrics"],
            }
        else:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }
        model_time = time.monotonic() - start_time
        attention_metrics = outputs.get("attention_metrics")
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if attention_metrics is not None:
            outputs["attention_metrics"] = jax.tree.map(lambda x: np.asarray(x[0, ...]), attention_metrics)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
