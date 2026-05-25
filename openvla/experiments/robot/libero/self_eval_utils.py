"""Shared helpers for LIBERO / LIBERO-PRO self-evaluation scoring."""

from typing import Iterable, List, Optional, Sequence

import numpy as np

from experiments.robot.libero.attention_mad_utils import parse_attention_eval_ratios
from experiments.robot.libero.mae_metric_naming import (
    DEFAULT_MAE_D_OUTPUT,
    MAE_C_METHOD,
    MAE_D_MODE,
    MAE_NAMING_NOTE,
    is_mae_d_self_eval_mode,
    normalize_attention_method,
    normalize_self_eval_mode,
)


def compute_step_token_consistency(token_id_samples: Sequence[np.ndarray]) -> Optional[float]:
    """
    Consistency of K sampled discrete action-token vectors at one decision step.
    For each of the 7 action dims, score = (max duplicate count) / K, then mean over dims.
    """
    if not token_id_samples:
        return None
    if len(token_id_samples) == 1:
        return 1.0
    stacked = np.stack([np.asarray(x, dtype=np.int64).reshape(-1) for x in token_id_samples], axis=0)
    if stacked.ndim != 2 or stacked.shape[0] < 2:
        return None
    dim_scores = []
    for dim_idx in range(stacked.shape[1]):
        _, counts = np.unique(stacked[:, dim_idx], return_counts=True)
        dim_scores.append(float(np.max(counts)) / float(stacked.shape[0]))
    return float(np.mean(dim_scores))


def aggregate_episode_token_consistency(step_scores: Iterable[float]) -> Optional[float]:
    scores = [float(x) for x in step_scores if x is not None]
    if not scores:
        return None
    return float(np.mean(scores))


def configure_self_eval(cfg) -> None:
    """Validate and wire self-eval mode flags on the eval config object."""
    if not getattr(cfg, "enable_self_eval", False):
        return

    print(f"[OpenVLA self-eval] {MAE_NAMING_NOTE}")

    mode = normalize_self_eval_mode(cfg.self_eval_mode)
    if mode not in {"consistency", "output_stats", MAE_D_MODE}:
        raise ValueError(
            f"Unsupported self_eval_mode={cfg.self_eval_mode!r}. "
            f"Expected 'consistency', 'output_stats', or '{MAE_D_MODE}'."
        )
    cfg.self_eval_mode = mode

    method = normalize_attention_method(getattr(cfg, "attention_eval_method", MAE_D_MODE))
    if method == MAE_C_METHOD:
        raise ValueError(
            "OpenVLA self-eval supports MAE-D only (not MAE-C). "
            "Use starVLA PI with attention_eval_method=mae-c."
        )
    cfg.attention_eval_method = MAE_D_MODE

    if mode == "consistency":
        if cfg.consistency_repeats_per_init <= 0:
            raise ValueError(
                f"consistency_repeats_per_init must be > 0, got {cfg.consistency_repeats_per_init}"
            )
        if getattr(cfg, "action_sampling_temperature", 0.0) < 0:
            raise ValueError(
                f"action_sampling_temperature must be >= 0, got {cfg.action_sampling_temperature}"
            )
        consistency_method = str(getattr(cfg, "consistency_method", "token_sample")).strip().lower()
        if consistency_method not in {"token_sample", "outcome_repeat"}:
            raise ValueError(
                f"Unsupported consistency_method={cfg.consistency_method!r}. "
                "Expected 'token_sample' or 'outcome_repeat'."
            )
        cfg.consistency_method = consistency_method
        if consistency_method == "token_sample" and float(getattr(cfg, "action_sampling_temperature", 0.0)) <= 0:
            raise ValueError(
                "consistency_method=token_sample requires action_sampling_temperature > 0 "
                "so OpenVLA can sample different action tokens at each step."
            )
        cfg.track_output_stats = False
    elif is_mae_d_self_eval_mode(mode):
        cfg.track_output_stats = False
        cfg.save_attentions = False
        cfg.attention_eval_ratios_list = parse_attention_eval_ratios(
            getattr(cfg, "attention_eval_ratios", "0.01")
        )
    else:
        cfg.track_output_stats = True


def self_eval_json_filename(
    mode: str,
    *,
    attention_eval_output_name: str = DEFAULT_MAE_D_OUTPUT,
) -> str:
    if mode == "consistency":
        return "self_eval_consistency_scores.json"
    if is_mae_d_self_eval_mode(mode):
        return attention_eval_output_name
    return "self_eval_output_stats_scores.json"
