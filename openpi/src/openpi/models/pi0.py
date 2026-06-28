import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.attention_num_layers = paligemma_config.depth
        self.attention_num_heads = paligemma_config.num_heads
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        return self._sample_actions(rng, observation, num_steps=num_steps, noise=noise, collect_attention_metrics=False)

    def sample_actions_with_attention_metrics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        metric_mode: str = "both",
        metric_ratios: tuple[float, ...] = (0.01, 0.05, 0.10, 0.50),
        tts_mode: str = "none",
        num_candidates: int = 4,
        selection_ratio: float = 0.10,
        branch_ratio: float = 0.40,
        branch_noise_scale: float = 0.10,
        tts_score_mode: str = "mae",
        flow_mg_mask: str = "language",
        flow_mg_steps: tuple[int, ...] = (4, 7, 9),
    ):
        if metric_mode not in ("mac", "mad", "both"):
            raise ValueError(f"Unsupported attention metric mode: {metric_mode}")
        if tts_mode not in ("none", "independent", "branch"):
            raise ValueError(f"Unsupported tts_mode: {tts_mode}")
        if tts_score_mode not in ("mae", "velocity_diff", "mae_diff", "mae_velocity_diff"):
            raise ValueError(f"Unsupported tts_score_mode: {tts_score_mode}")
        if flow_mg_mask not in ("language", "vision", "language_vision"):
            raise ValueError(f"Unsupported flow_mg_mask: {flow_mg_mask}")
        if not flow_mg_steps:
            raise ValueError("flow_mg_steps must contain at least one ODE step index")
        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
        if not metric_ratios or any(ratio <= 0 or ratio > 1 for ratio in metric_ratios):
            raise ValueError(f"Attention metric ratios must be in (0, 1], got: {metric_ratios}")
        if selection_ratio <= 0 or selection_ratio > 1:
            raise ValueError(f"selection_ratio must be in (0, 1], got: {selection_ratio}")
        if selection_ratio not in metric_ratios:
            raise ValueError(f"selection_ratio={selection_ratio} must appear in metric_ratios={metric_ratios}")
        if not 0.0 <= branch_ratio < 1.0:
            raise ValueError(f"branch_ratio must be in [0, 1), got: {branch_ratio}")
        if branch_noise_scale < 0.0:
            raise ValueError(f"branch_noise_scale must be non-negative, got: {branch_noise_scale}")
        if tts_mode == "independent":
            if noise is not None:
                raise ValueError("Explicit noise is not supported with tts_mode='independent'.")
            return self._sample_actions_independent_best_of_n(
                rng,
                observation,
                num_steps=num_steps,
                num_candidates=num_candidates,
                metric_mode=metric_mode,
                metric_ratios=metric_ratios,
                selection_ratio=selection_ratio,
                tts_score_mode=tts_score_mode,
                flow_mg_mask=flow_mg_mask,
                flow_mg_steps=flow_mg_steps,
            )
        if tts_mode == "branch":
            if noise is not None:
                raise ValueError("Explicit noise is not supported with tts_mode='branch'.")
            return self._sample_actions_ancestral_branching(
                rng,
                observation,
                num_steps=num_steps,
                num_candidates=num_candidates,
                metric_mode=metric_mode,
                metric_ratios=metric_ratios,
                selection_ratio=selection_ratio,
                branch_ratio=branch_ratio,
                branch_noise_scale=branch_noise_scale,
                tts_score_mode=tts_score_mode,
                flow_mg_mask=flow_mg_mask,
                flow_mg_steps=flow_mg_steps,
            )
        return self._sample_actions(
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
            collect_attention_metrics=True,
            metric_mode=metric_mode,
            metric_ratios=metric_ratios,
        )

    def _prepare_sampling_context(self, observation: _model.Observation):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        prompt_length = observation.tokenized_prompt.shape[1] if observation.tokenized_prompt is not None else 0
        return {
            "prefix_tokens": prefix_tokens,
            "prefix_mask": prefix_mask,
            "kv_cache": kv_cache,
            "visual_prefix_length": prefix_tokens.shape[1] - prompt_length,
        }

    def _flow_velocity(
        self,
        observation: _model.Observation,
        x_t: at.Float[at.Array, "b ah ad"],
        time: at.Float[at.Array, ""],
        *,
        prefix_tokens: at.Float[at.Array, "b p emb"],
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache,
    ):
        batch_size = x_t.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def _flow_step_with_attention(
        self,
        observation: _model.Observation,
        x_t: at.Float[at.Array, "b ah ad"],
        time: at.Float[at.Array, ""],
        *,
        prefix_tokens: at.Float[at.Array, "b p emb"],
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache,
        visual_prefix_length: int,
    ):
        batch_size = x_t.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        metric_action_queries = min(8, self.action_horizon)
        query_is_action = jnp.arange(suffix_tokens.shape[1]) >= (suffix_tokens.shape[1] - metric_action_queries)
        key_is_visual = jnp.arange(full_attn_mask.shape[-1]) < visual_prefix_length
        attention_stats_mask = query_is_action[None, :, None] & key_is_visual[None, None, :] & full_attn_mask
        (prefix_out, suffix_out), _, layer_head_entropy = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            attention_stats_mask=attention_stats_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            method="with_attention_stats",
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :]), layer_head_entropy

    def _make_flow_mg_masked_observation(
        self,
        observation: _model.Observation,
        *,
        flow_mg_mask: str,
    ) -> _model.Observation:
        mask_language = "language" in flow_mg_mask
        mask_vision = "vision" in flow_mg_mask

        tokenized_prompt_mask = observation.tokenized_prompt_mask
        if mask_language and tokenized_prompt_mask is not None:
            tokenized_prompt_mask = jnp.zeros_like(tokenized_prompt_mask, dtype=jnp.bool_)

        images = observation.images
        if mask_vision:
            images = {name: jnp.zeros_like(image) for name, image in observation.images.items()}

        return _model.Observation(
            images=images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )

    def _repeat_observation(self, observation: _model.Observation, *, repeats: int) -> _model.Observation:
        def repeat_or_none(value):
            return None if value is None else jnp.repeat(value, repeats, axis=0)

        return _model.Observation(
            images={name: jnp.repeat(image, repeats, axis=0) for name, image in observation.images.items()},
            image_masks={name: jnp.repeat(mask, repeats, axis=0) for name, mask in observation.image_masks.items()},
            state=jnp.repeat(observation.state, repeats, axis=0),
            tokenized_prompt=repeat_or_none(observation.tokenized_prompt),
            tokenized_prompt_mask=repeat_or_none(observation.tokenized_prompt_mask),
            token_ar_mask=repeat_or_none(observation.token_ar_mask),
            token_loss_mask=repeat_or_none(observation.token_loss_mask),
        )

    def _compute_attention_metrics(
        self,
        entropy: at.Float[at.Array, "b l h"],
        *,
        metric_mode: str,
        metric_ratios: tuple[float, ...],
    ):
        sorted_entropy = jnp.sort(entropy, axis=-1)
        metrics = {}
        for ratio in metric_ratios:
            k = max(1, int(np.ceil(self.attention_num_heads * ratio)))
            ratio_tag = round(ratio * 100)
            if metric_mode in ("mac", "both"):
                mac_layers = jnp.mean(sorted_entropy[..., :k], axis=-1)
                metrics[f"mac_bottom{ratio_tag}"] = jnp.mean(mac_layers, axis=-1)
                metrics[f"mac_bottom{ratio_tag}_layers"] = mac_layers
            if metric_mode in ("mad", "both"):
                mad_layers = -jnp.mean(sorted_entropy[..., -k:], axis=-1)
                metrics[f"mad_top{ratio_tag}"] = jnp.mean(mad_layers, axis=-1)
                metrics[f"mad_top{ratio_tag}_layers"] = mad_layers
        return metrics

    def _compute_independent_attention_metrics(
        self,
        entropy: at.Float[at.Array, "b n k l h"],
        *,
        metric_mode: str,
        metric_ratios: tuple[float, ...],
    ):
        sorted_entropy = jnp.sort(entropy, axis=-1)
        metrics = {}
        for ratio in metric_ratios:
            k = max(1, int(np.ceil(self.attention_num_heads * ratio)))
            ratio_tag = round(ratio * 100)
            bottom_layers = jnp.mean(sorted_entropy[..., :k], axis=-1)
            mae_scores = jnp.mean(bottom_layers, axis=(-2, -1))
            metrics[f"mae_bottom{ratio_tag}"] = mae_scores
            metrics[f"mae_bottom{ratio_tag}_steps_layers"] = bottom_layers
            if metric_mode in ("mac", "both"):
                metrics[f"mac_bottom{ratio_tag}"] = mae_scores
                metrics[f"mac_bottom{ratio_tag}_steps_layers"] = bottom_layers
            if metric_mode in ("mad", "both"):
                top_layers = -jnp.mean(sorted_entropy[..., -k:], axis=-1)
                metrics[f"mad_top{ratio_tag}"] = jnp.mean(top_layers, axis=(-2, -1))
                metrics[f"mad_top{ratio_tag}_steps_layers"] = top_layers
        return metrics

    def _sample_actions_independent_best_of_n(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""],
        num_candidates: int,
        metric_mode: str,
        metric_ratios: tuple[float, ...],
        selection_ratio: float,
        tts_score_mode: str,
        flow_mg_mask: str,
        flow_mg_steps: tuple[int, ...],
    ):
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        context = self._prepare_sampling_context(observation)

        initial_noise = jax.random.normal(
            rng,
            (batch_size, num_candidates, self.action_horizon, self.action_dim),
        )
        x_t = einops.rearrange(initial_noise, "b n ah ad -> (b n) ah ad")
        candidate_observation = self._repeat_observation(observation, repeats=num_candidates)
        candidate_prefix_tokens = jnp.repeat(context["prefix_tokens"], num_candidates, axis=0)
        candidate_prefix_mask = jnp.repeat(context["prefix_mask"], num_candidates, axis=0)
        candidate_kv_cache = jax.tree.map(lambda value: jnp.repeat(value, num_candidates, axis=1), context["kv_cache"])

        if tts_score_mode in ("velocity_diff", "mae_diff", "mae_velocity_diff"):
            masked_observation = self._make_flow_mg_masked_observation(observation, flow_mg_mask=flow_mg_mask)
            masked_context = self._prepare_sampling_context(masked_observation)
            masked_candidate_observation = self._repeat_observation(masked_observation, repeats=num_candidates)
            masked_candidate_prefix_tokens = jnp.repeat(masked_context["prefix_tokens"], num_candidates, axis=0)
            masked_candidate_prefix_mask = jnp.repeat(masked_context["prefix_mask"], num_candidates, axis=0)
            masked_candidate_kv_cache = jax.tree.map(
                lambda value: jnp.repeat(value, num_candidates, axis=1), masked_context["kv_cache"]
            )
            flow_mg_steps_array = jnp.asarray(flow_mg_steps, dtype=jnp.int32)

            def scan_step(carry, _):
                x_t, time, step_index = carry
                velocity, layer_head_entropy = self._flow_step_with_attention(
                    candidate_observation,
                    x_t,
                    time,
                    prefix_tokens=candidate_prefix_tokens,
                    prefix_mask=candidate_prefix_mask,
                    kv_cache=candidate_kv_cache,
                    visual_prefix_length=context["visual_prefix_length"],
                )

                def compute_masked(_):
                    masked_velocity, masked_layer_head_entropy = self._flow_step_with_attention(
                        masked_candidate_observation,
                        x_t,
                        time,
                        prefix_tokens=masked_candidate_prefix_tokens,
                        prefix_mask=masked_candidate_prefix_mask,
                        kv_cache=masked_candidate_kv_cache,
                        visual_prefix_length=masked_context["visual_prefix_length"],
                    )
                    velocity_score = jnp.mean(jnp.square(velocity - masked_velocity), axis=(-2, -1)).astype(
                        jnp.float32
                    )
                    return velocity_score, masked_layer_head_entropy

                def compute_velocity_score(_):
                    masked_velocity = self._flow_velocity(
                        masked_candidate_observation,
                        x_t,
                        time,
                        prefix_tokens=masked_candidate_prefix_tokens,
                        prefix_mask=masked_candidate_prefix_mask,
                        kv_cache=masked_candidate_kv_cache,
                    )
                    return jnp.mean(jnp.square(velocity - masked_velocity), axis=(-2, -1)).astype(jnp.float32)

                use_step = jnp.any(flow_mg_steps_array == step_index)
                if tts_score_mode in ("mae_diff", "mae_velocity_diff"):
                    score_step, masked_entropy = jax.lax.cond(
                        use_step,
                        compute_masked,
                        lambda _: (
                            jnp.zeros(x_t.shape[0], dtype=jnp.float32),
                            jnp.zeros_like(layer_head_entropy),
                        ),
                        operand=None,
                    )
                else:
                    score_step = jax.lax.cond(
                        use_step,
                        compute_velocity_score,
                        lambda _: jnp.zeros(x_t.shape[0], dtype=jnp.float32),
                        operand=None,
                    )
                    masked_entropy = jnp.zeros_like(layer_head_entropy)
                return (
                    x_t + dt * velocity,
                    time + dt,
                    step_index + 1,
                ), (layer_head_entropy, score_step, use_step, masked_entropy)

            (x_0, _, _), (entropy_steps, flow_mg_score_steps, flow_mg_used_steps, masked_entropy_steps) = jax.lax.scan(
                scan_step,
                (x_t, jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(0, dtype=jnp.int32)),
                xs=None,
                length=num_steps,
            )
        else:
            def scan_step(carry, _):
                x_t, time = carry
                velocity, layer_head_entropy = self._flow_step_with_attention(
                    candidate_observation,
                    x_t,
                    time,
                    prefix_tokens=candidate_prefix_tokens,
                    prefix_mask=candidate_prefix_mask,
                    kv_cache=candidate_kv_cache,
                    visual_prefix_length=context["visual_prefix_length"],
                )
                return (x_t + dt * velocity, time + dt), layer_head_entropy

            (x_0, _), entropy_steps = jax.lax.scan(
                scan_step,
                (x_t, jnp.asarray(1.0, dtype=jnp.float32)),
                xs=None,
                length=num_steps,
            )
        entropy = jnp.transpose(entropy_steps, (2, 0, 1, 3))
        entropy = einops.rearrange(entropy, "(b n) k l h -> b n k l h", b=batch_size, n=num_candidates)
        candidate_actions = einops.rearrange(x_0, "(b n) ah ad -> b n ah ad", b=batch_size, n=num_candidates)

        metrics = self._compute_independent_attention_metrics(
            entropy,
            metric_mode=metric_mode,
            metric_ratios=metric_ratios,
        )
        selection_tag = round(selection_ratio * 100)
        if tts_score_mode in ("velocity_diff", "mae_velocity_diff"):
            flow_mg_scores = einops.rearrange(
                flow_mg_score_steps,
                "k (b n) -> b n k",
                b=batch_size,
                n=num_candidates,
            )
            used = flow_mg_used_steps.astype(flow_mg_scores.dtype)
            flow_mg_velocity_diff = jnp.sum(flow_mg_scores * used[None, None, :], axis=-1) / jnp.maximum(
                jnp.sum(used), 1.0
            )
            metrics["flow_mg_velocity_diff"] = flow_mg_velocity_diff
            metrics["flow_mg_num_scored_steps"] = jnp.broadcast_to(
                jnp.sum(flow_mg_used_steps).astype(jnp.int32), (batch_size,)
            )
        if tts_score_mode in ("mae_diff", "mae_velocity_diff"):
            masked_entropy = jnp.transpose(masked_entropy_steps, (2, 0, 1, 3))
            masked_entropy = einops.rearrange(
                masked_entropy,
                "(b n) k l h -> b n k l h",
                b=batch_size,
                n=num_candidates,
            )
            used = flow_mg_used_steps.astype(masked_entropy.dtype)
            masked_entropy = masked_entropy * used[None, None, :, None, None]
            full_entropy_for_diff = entropy * used[None, None, :, None, None]
            mae_diff_metrics = self._compute_independent_attention_metrics(
                full_entropy_for_diff,
                metric_mode=metric_mode,
                metric_ratios=metric_ratios,
            )
            masked_mae_metrics = self._compute_independent_attention_metrics(
                masked_entropy,
                metric_mode=metric_mode,
                metric_ratios=metric_ratios,
            )
            used_count = jnp.maximum(jnp.sum(flow_mg_used_steps).astype(masked_entropy.dtype), 1.0)
            for ratio in metric_ratios:
                ratio_tag = round(ratio * 100)
                full_mae = mae_diff_metrics[f"mae_bottom{ratio_tag}"] * (num_steps / used_count)
                masked_mae = masked_mae_metrics[f"mae_bottom{ratio_tag}"] * (num_steps / used_count)
                metrics[f"flow_mg_mae_full_bottom{ratio_tag}"] = full_mae
                metrics[f"flow_mg_mae_masked_bottom{ratio_tag}"] = masked_mae
                metrics[f"flow_mg_mae_diff_bottom{ratio_tag}"] = full_mae - masked_mae
            metrics["flow_mg_num_scored_steps"] = jnp.broadcast_to(
                jnp.sum(flow_mg_used_steps).astype(jnp.int32), (batch_size,)
            )
        if tts_score_mode == "velocity_diff":
            selection_scores = flow_mg_velocity_diff
        elif tts_score_mode == "mae_diff":
            selection_scores = metrics[f"flow_mg_mae_diff_bottom{selection_tag}"]
        elif tts_score_mode == "mae_velocity_diff":
            mae_diff_scores = metrics[f"flow_mg_mae_diff_bottom{selection_tag}"]
            mae_diff_norm = (mae_diff_scores - jnp.min(mae_diff_scores, axis=1, keepdims=True)) / jnp.maximum(
                jnp.max(mae_diff_scores, axis=1, keepdims=True) - jnp.min(mae_diff_scores, axis=1, keepdims=True),
                1e-6,
            )
            velocity_diff_norm = (
                flow_mg_velocity_diff - jnp.min(flow_mg_velocity_diff, axis=1, keepdims=True)
            ) / jnp.maximum(
                jnp.max(flow_mg_velocity_diff, axis=1, keepdims=True)
                - jnp.min(flow_mg_velocity_diff, axis=1, keepdims=True),
                1e-6,
            )
            metrics["flow_mg_mae_diff_norm"] = mae_diff_norm
            metrics["flow_mg_velocity_diff_norm"] = velocity_diff_norm
            selection_scores = 0.5 * mae_diff_norm + 0.5 * velocity_diff_norm
            metrics["flow_mg_mae_velocity_diff"] = selection_scores
        else:
            selection_scores = metrics[f"mae_bottom{selection_tag}"]
        best_indices = jnp.argmax(selection_scores, axis=1)
        selected_actions = jnp.take_along_axis(
            candidate_actions,
            best_indices[:, None, None, None],
            axis=1,
        )[:, 0]
        metrics["tts_best_candidate_index"] = best_indices
        metrics["tts_selection_score"] = jnp.take_along_axis(selection_scores, best_indices[:, None], axis=1)[:, 0]
        metrics["tts_score_mode_velocity_diff"] = jnp.full(
            (batch_size,), 1 if tts_score_mode == "velocity_diff" else 0, dtype=jnp.int32
        )
        metrics["tts_score_mode_mae_diff"] = jnp.full(
            (batch_size,), 1 if tts_score_mode == "mae_diff" else 0, dtype=jnp.int32
        )
        metrics["tts_score_mode_mae_velocity_diff"] = jnp.full(
            (batch_size,), 1 if tts_score_mode == "mae_velocity_diff" else 0, dtype=jnp.int32
        )
        metrics["tts_num_candidates"] = jnp.full((batch_size,), num_candidates, dtype=jnp.int32)
        if tts_score_mode == "velocity_diff":
            metrics["flow_mg_velocity_diff_selected"] = metrics["tts_selection_score"]
        if tts_score_mode == "mae_diff":
            metrics["flow_mg_mae_diff_selected"] = metrics["tts_selection_score"]
        if tts_score_mode == "mae_velocity_diff":
            metrics["flow_mg_mae_velocity_diff_selected"] = metrics["tts_selection_score"]
        for ratio in metric_ratios:
            ratio_tag = round(ratio * 100)
            mae_scores = metrics[f"mae_bottom{ratio_tag}"]
            metrics[f"mae_bottom{ratio_tag}_selected"] = jnp.take_along_axis(
                mae_scores,
                best_indices[:, None],
                axis=1,
            )[:, 0]
            if metric_mode in ("mac", "both"):
                metrics[f"mac_bottom{ratio_tag}_selected"] = metrics[f"mae_bottom{ratio_tag}_selected"]
            if metric_mode in ("mad", "both"):
                mad_scores = metrics[f"mad_top{ratio_tag}"]
                metrics[f"mad_top{ratio_tag}_selected"] = jnp.take_along_axis(
                    mad_scores,
                    best_indices[:, None],
                    axis=1,
                )[:, 0]
        return {"actions": selected_actions, "attention_metrics": metrics}

    def _sample_actions_ancestral_branching(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""],
        num_candidates: int,
        metric_mode: str,
        metric_ratios: tuple[float, ...],
        selection_ratio: float,
        branch_ratio: float,
        branch_noise_scale: float,
        tts_score_mode: str,
        flow_mg_mask: str,
        flow_mg_steps: tuple[int, ...],
    ):
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        branch_step = min(num_steps - 1, max(0, round(branch_ratio * num_steps)))
        context = self._prepare_sampling_context(observation)

        rng_initial, rng_branch = jax.random.split(rng)
        x_initial = jax.random.normal(
            rng_initial,
            (batch_size, self.action_horizon, self.action_dim),
        )

        def shared_step(carry, _):
            x_t, time = carry
            velocity, _ = self._flow_step_with_attention(
                observation,
                x_t,
                time,
                prefix_tokens=context["prefix_tokens"],
                prefix_mask=context["prefix_mask"],
                kv_cache=context["kv_cache"],
                visual_prefix_length=context["visual_prefix_length"],
            )
            return (x_t + dt * velocity, time + dt), None

        (x_shared, shared_time), _ = jax.lax.scan(
            shared_step,
            (x_initial, jnp.asarray(1.0, dtype=jnp.float32)),
            xs=None,
            length=branch_step,
        )

        x_branches = jnp.repeat(x_shared[:, None, :, :], num_candidates, axis=1)
        epsilon = jax.random.normal(rng_branch, x_branches.shape)
        latent_std = jnp.std(x_shared, axis=(-2, -1), keepdims=True)[:, None, :, :]
        sigma = branch_noise_scale * jnp.maximum(latent_std, 1e-6)
        x_t = einops.rearrange(x_branches + sigma * epsilon, "b n ah ad -> (b n) ah ad")

        candidate_observation = self._repeat_observation(observation, repeats=num_candidates)
        candidate_prefix_tokens = jnp.repeat(context["prefix_tokens"], num_candidates, axis=0)
        candidate_prefix_mask = jnp.repeat(context["prefix_mask"], num_candidates, axis=0)
        candidate_kv_cache = jax.tree.map(lambda value: jnp.repeat(value, num_candidates, axis=1), context["kv_cache"])

        remaining_steps = num_steps - branch_step
        if tts_score_mode in ("velocity_diff", "mae_diff", "mae_velocity_diff"):
            masked_observation = self._make_flow_mg_masked_observation(observation, flow_mg_mask=flow_mg_mask)
            masked_context = self._prepare_sampling_context(masked_observation)
            masked_candidate_observation = self._repeat_observation(masked_observation, repeats=num_candidates)
            masked_candidate_prefix_tokens = jnp.repeat(masked_context["prefix_tokens"], num_candidates, axis=0)
            masked_candidate_prefix_mask = jnp.repeat(masked_context["prefix_mask"], num_candidates, axis=0)
            masked_candidate_kv_cache = jax.tree.map(
                lambda value: jnp.repeat(value, num_candidates, axis=1), masked_context["kv_cache"]
            )
            flow_mg_steps_array = jnp.asarray(flow_mg_steps, dtype=jnp.int32)

            def candidate_step(carry, _):
                x_t, time, step_index = carry
                velocity, layer_head_entropy = self._flow_step_with_attention(
                    candidate_observation,
                    x_t,
                    time,
                    prefix_tokens=candidate_prefix_tokens,
                    prefix_mask=candidate_prefix_mask,
                    kv_cache=candidate_kv_cache,
                    visual_prefix_length=context["visual_prefix_length"],
                )

                def compute_masked(_):
                    masked_velocity, masked_layer_head_entropy = self._flow_step_with_attention(
                        masked_candidate_observation,
                        x_t,
                        time,
                        prefix_tokens=masked_candidate_prefix_tokens,
                        prefix_mask=masked_candidate_prefix_mask,
                        kv_cache=masked_candidate_kv_cache,
                        visual_prefix_length=masked_context["visual_prefix_length"],
                    )
                    velocity_score = jnp.mean(jnp.square(velocity - masked_velocity), axis=(-2, -1)).astype(
                        jnp.float32
                    )
                    return velocity_score, masked_layer_head_entropy

                def compute_velocity_score(_):
                    masked_velocity = self._flow_velocity(
                        masked_candidate_observation,
                        x_t,
                        time,
                        prefix_tokens=masked_candidate_prefix_tokens,
                        prefix_mask=masked_candidate_prefix_mask,
                        kv_cache=masked_candidate_kv_cache,
                    )
                    return jnp.mean(jnp.square(velocity - masked_velocity), axis=(-2, -1)).astype(jnp.float32)

                use_step = jnp.any(flow_mg_steps_array == step_index)
                if tts_score_mode in ("mae_diff", "mae_velocity_diff"):
                    score_step, masked_entropy = jax.lax.cond(
                        use_step,
                        compute_masked,
                        lambda _: (
                            jnp.zeros(x_t.shape[0], dtype=jnp.float32),
                            jnp.zeros_like(layer_head_entropy),
                        ),
                        operand=None,
                    )
                else:
                    score_step = jax.lax.cond(
                        use_step,
                        compute_velocity_score,
                        lambda _: jnp.zeros(x_t.shape[0], dtype=jnp.float32),
                        operand=None,
                    )
                    masked_entropy = jnp.zeros_like(layer_head_entropy)
                return (
                    x_t + dt * velocity,
                    time + dt,
                    step_index + 1,
                ), (layer_head_entropy, score_step, use_step, masked_entropy)

            (x_0, _, _), (entropy_steps, flow_mg_score_steps, flow_mg_used_steps, masked_entropy_steps) = jax.lax.scan(
                candidate_step,
                (x_t, shared_time, jnp.asarray(branch_step, dtype=jnp.int32)),
                xs=None,
                length=remaining_steps,
            )
        else:
            def candidate_step(carry, _):
                x_t, time = carry
                velocity, layer_head_entropy = self._flow_step_with_attention(
                    candidate_observation,
                    x_t,
                    time,
                    prefix_tokens=candidate_prefix_tokens,
                    prefix_mask=candidate_prefix_mask,
                    kv_cache=candidate_kv_cache,
                    visual_prefix_length=context["visual_prefix_length"],
                )
                return (x_t + dt * velocity, time + dt), layer_head_entropy

            (x_0, _), entropy_steps = jax.lax.scan(
                candidate_step,
                (x_t, shared_time),
                xs=None,
                length=remaining_steps,
            )
        entropy = jnp.transpose(entropy_steps, (2, 0, 1, 3))
        entropy = einops.rearrange(entropy, "(b n) k l h -> b n k l h", b=batch_size, n=num_candidates)
        candidate_actions = einops.rearrange(x_0, "(b n) ah ad -> b n ah ad", b=batch_size, n=num_candidates)

        metrics = self._compute_independent_attention_metrics(
            entropy,
            metric_mode=metric_mode,
            metric_ratios=metric_ratios,
        )
        selection_tag = round(selection_ratio * 100)
        if tts_score_mode in ("velocity_diff", "mae_velocity_diff"):
            flow_mg_scores = einops.rearrange(
                flow_mg_score_steps,
                "k (b n) -> b n k",
                b=batch_size,
                n=num_candidates,
            )
            used = flow_mg_used_steps.astype(flow_mg_scores.dtype)
            flow_mg_velocity_diff = jnp.sum(flow_mg_scores * used[None, None, :], axis=-1) / jnp.maximum(
                jnp.sum(used), 1.0
            )
            metrics["flow_mg_velocity_diff"] = flow_mg_velocity_diff
            metrics["flow_mg_num_scored_steps"] = jnp.broadcast_to(
                jnp.sum(flow_mg_used_steps).astype(jnp.int32), (batch_size,)
            )
        if tts_score_mode in ("mae_diff", "mae_velocity_diff"):
            masked_entropy = jnp.transpose(masked_entropy_steps, (2, 0, 1, 3))
            masked_entropy = einops.rearrange(
                masked_entropy,
                "(b n) k l h -> b n k l h",
                b=batch_size,
                n=num_candidates,
            )
            used = flow_mg_used_steps.astype(masked_entropy.dtype)
            masked_entropy = masked_entropy * used[None, None, :, None, None]
            full_entropy_for_diff = entropy * used[None, None, :, None, None]
            mae_diff_metrics = self._compute_independent_attention_metrics(
                full_entropy_for_diff,
                metric_mode=metric_mode,
                metric_ratios=metric_ratios,
            )
            masked_mae_metrics = self._compute_independent_attention_metrics(
                masked_entropy,
                metric_mode=metric_mode,
                metric_ratios=metric_ratios,
            )
            used_count = jnp.maximum(jnp.sum(flow_mg_used_steps).astype(masked_entropy.dtype), 1.0)
            for ratio in metric_ratios:
                ratio_tag = round(ratio * 100)
                full_mae = mae_diff_metrics[f"mae_bottom{ratio_tag}"] * (remaining_steps / used_count)
                masked_mae = masked_mae_metrics[f"mae_bottom{ratio_tag}"] * (remaining_steps / used_count)
                metrics[f"flow_mg_mae_full_bottom{ratio_tag}"] = full_mae
                metrics[f"flow_mg_mae_masked_bottom{ratio_tag}"] = masked_mae
                metrics[f"flow_mg_mae_diff_bottom{ratio_tag}"] = full_mae - masked_mae
            metrics["flow_mg_num_scored_steps"] = jnp.broadcast_to(
                jnp.sum(flow_mg_used_steps).astype(jnp.int32), (batch_size,)
            )
        if tts_score_mode == "velocity_diff":
            selection_scores = flow_mg_velocity_diff
        elif tts_score_mode == "mae_diff":
            selection_scores = metrics[f"flow_mg_mae_diff_bottom{selection_tag}"]
        elif tts_score_mode == "mae_velocity_diff":
            mae_diff_scores = metrics[f"flow_mg_mae_diff_bottom{selection_tag}"]
            mae_diff_norm = (mae_diff_scores - jnp.min(mae_diff_scores, axis=1, keepdims=True)) / jnp.maximum(
                jnp.max(mae_diff_scores, axis=1, keepdims=True) - jnp.min(mae_diff_scores, axis=1, keepdims=True),
                1e-6,
            )
            velocity_diff_norm = (
                flow_mg_velocity_diff - jnp.min(flow_mg_velocity_diff, axis=1, keepdims=True)
            ) / jnp.maximum(
                jnp.max(flow_mg_velocity_diff, axis=1, keepdims=True)
                - jnp.min(flow_mg_velocity_diff, axis=1, keepdims=True),
                1e-6,
            )
            metrics["flow_mg_mae_diff_norm"] = mae_diff_norm
            metrics["flow_mg_velocity_diff_norm"] = velocity_diff_norm
            selection_scores = 0.5 * mae_diff_norm + 0.5 * velocity_diff_norm
            metrics["flow_mg_mae_velocity_diff"] = selection_scores
        else:
            selection_scores = metrics[f"mae_bottom{selection_tag}"]
        best_indices = jnp.argmax(selection_scores, axis=1)
        selected_actions = jnp.take_along_axis(
            candidate_actions,
            best_indices[:, None, None, None],
            axis=1,
        )[:, 0]
        metrics["tts_best_candidate_index"] = best_indices
        metrics["tts_selection_score"] = jnp.take_along_axis(selection_scores, best_indices[:, None], axis=1)[:, 0]
        metrics["tts_score_mode_velocity_diff"] = jnp.full(
            (batch_size,), 1 if tts_score_mode == "velocity_diff" else 0, dtype=jnp.int32
        )
        metrics["tts_score_mode_mae_diff"] = jnp.full(
            (batch_size,), 1 if tts_score_mode == "mae_diff" else 0, dtype=jnp.int32
        )
        metrics["tts_score_mode_mae_velocity_diff"] = jnp.full(
            (batch_size,), 1 if tts_score_mode == "mae_velocity_diff" else 0, dtype=jnp.int32
        )
        metrics["tts_num_candidates"] = jnp.full((batch_size,), num_candidates, dtype=jnp.int32)
        metrics["tts_branch_step"] = jnp.full((batch_size,), branch_step, dtype=jnp.int32)
        metrics["tts_branch_ratio"] = jnp.full((batch_size,), branch_ratio, dtype=jnp.float32)
        metrics["tts_branch_noise_scale"] = jnp.full((batch_size,), branch_noise_scale, dtype=jnp.float32)
        if tts_score_mode == "velocity_diff":
            metrics["flow_mg_velocity_diff_selected"] = metrics["tts_selection_score"]
        if tts_score_mode == "mae_diff":
            metrics["flow_mg_mae_diff_selected"] = metrics["tts_selection_score"]
        if tts_score_mode == "mae_velocity_diff":
            metrics["flow_mg_mae_velocity_diff_selected"] = metrics["tts_selection_score"]
        for ratio in metric_ratios:
            ratio_tag = round(ratio * 100)
            mae_scores = metrics[f"mae_bottom{ratio_tag}"]
            metrics[f"mae_bottom{ratio_tag}_selected"] = jnp.take_along_axis(
                mae_scores,
                best_indices[:, None],
                axis=1,
            )[:, 0]
            if metric_mode in ("mac", "both"):
                metrics[f"mac_bottom{ratio_tag}_selected"] = metrics[f"mae_bottom{ratio_tag}_selected"]
            if metric_mode in ("mad", "both"):
                mad_scores = metrics[f"mad_top{ratio_tag}"]
                metrics[f"mad_top{ratio_tag}_selected"] = jnp.take_along_axis(
                    mad_scores,
                    best_indices[:, None],
                    axis=1,
                )[:, 0]
        return {"actions": selected_actions, "attention_metrics": metrics}

    def _sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        collect_attention_metrics: bool,
        metric_mode: str = "both",
        metric_ratios: tuple[float, ...] = (0.01, 0.05, 0.10, 0.50),
    ):
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        context = self._prepare_sampling_context(observation)
        prefix_tokens = context["prefix_tokens"]
        prefix_mask = context["prefix_mask"]
        kv_cache = context["kv_cache"]
        visual_prefix_length = context["visual_prefix_length"]
        num_layers = self.attention_num_layers
        num_heads = self.attention_num_heads

        def step(carry):
            if collect_attention_metrics:
                x_t, time, _ = carry
            else:
                x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            if collect_attention_metrics:
                metric_action_queries = min(8, self.action_horizon)
                query_is_action = jnp.arange(suffix_tokens.shape[1]) >= (suffix_tokens.shape[1] - metric_action_queries)
                key_is_visual = jnp.arange(full_attn_mask.shape[-1]) < visual_prefix_length
                attention_stats_mask = (
                    query_is_action[None, :, None] & key_is_visual[None, None, :] & full_attn_mask & (time <= -1.5 * dt)
                )
                (prefix_out, suffix_out), _, layer_head_entropy = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    attention_stats_mask=attention_stats_mask,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    method="with_attention_stats",
                )
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            if collect_attention_metrics:
                return x_t + dt * v_t, time + dt, layer_head_entropy
            return x_t + dt * v_t, time + dt

        def cond(carry):
            time = carry[1]
            # robust to floating-point error
            return time >= -dt / 2

        if not collect_attention_metrics:
            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            return x_0

        initial_entropy = jnp.zeros((num_layers, batch_size, num_heads), dtype=jnp.float32)
        x_0, _, layer_head_entropy = jax.lax.while_loop(cond, step, (noise, 1.0, initial_entropy))
        entropy = jnp.transpose(layer_head_entropy, (1, 0, 2))
        metrics = self._compute_attention_metrics(entropy, metric_mode=metric_mode, metric_ratios=metric_ratios)
        return {"actions": x_0, "attention_metrics": metrics}
