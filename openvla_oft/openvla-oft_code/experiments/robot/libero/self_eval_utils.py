"""Shared helpers for LIBERO / LIBERO-PRO self-evaluation scoring (OpenVLA-OFT)."""

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


def apply_action_noise(action, action_noise_std: float):
    """Add Gaussian noise to the first 6 action dims (gripper unchanged)."""
    action = np.asarray(action, dtype=np.float32).copy()
    if action_noise_std > 0 and action.shape[-1] >= 6:
        action[..., :6] = action[..., :6] + np.random.normal(
            0.0, action_noise_std, size=6
        ).astype(np.float32)
    return action


def configure_self_eval(cfg) -> None:
    """Validate and wire self-eval flags on the eval config object."""
    if not getattr(cfg, "enable_self_eval", False):
        return

    print(f"[OpenVLA-OFT self-eval] {MAE_NAMING_NOTE}")

    mode = normalize_self_eval_mode(cfg.self_eval_mode)
    if mode not in {"consistency", MAE_D_MODE}:
        raise ValueError(
            f"OpenVLA-OFT self-eval supports 'consistency' or '{MAE_D_MODE}', "
            f"got {cfg.self_eval_mode!r}."
        )
    cfg.self_eval_mode = mode

    method = normalize_attention_method(getattr(cfg, "attention_eval_method", MAE_D_MODE))
    if method == MAE_C_METHOD:
        raise ValueError(
            "OpenVLA-OFT self-eval supports MAE-D only (not MAE-C). Use starVLA PI for MAE-C."
        )
    cfg.attention_eval_method = MAE_D_MODE

    if is_mae_d_self_eval_mode(mode):
        cfg.save_attentions = False
        cfg.track_uncertainty = False
        cfg.attention_eval_ratios_list = parse_attention_eval_ratios(
            getattr(cfg, "attention_eval_ratios", "0.01")
        )
        return

    if cfg.consistency_repeats_per_init <= 0:
        raise ValueError(
            f"consistency_repeats_per_init must be > 0, got {cfg.consistency_repeats_per_init}"
        )

    action_noise_std = float(getattr(cfg, "action_noise_std", 0.0))
    if action_noise_std < 0:
        raise ValueError(f"action_noise_std must be >= 0, got {action_noise_std}")
    cfg.action_noise_std = action_noise_std

    cfg.save_attentions = False
    cfg.track_uncertainty = False


def self_eval_json_filename(
    mode: str = "consistency",
    *,
    attention_eval_output_name: str = DEFAULT_MAE_D_OUTPUT,
) -> str:
    if is_mae_d_self_eval_mode(mode):
        return attention_eval_output_name
    return "self_eval_consistency_scores.json"


def consistency_num_repeats(cfg) -> int:
    if getattr(cfg, "enable_self_eval", False) and cfg.self_eval_mode == "consistency":
        return int(cfg.consistency_repeats_per_init)
    return 1


def append_consistency_record(
    cfg,
    self_eval_records,
    self_eval_json_path: str,
    task_id: int,
    task_description: str,
    episode_idx: int,
    repeat_successes,
    num_repeats: int,
) -> float:
    from experiments.robot.openvla_utils import write_self_eval_json

    success_count = int(sum(repeat_successes))
    consistency_score = float(success_count / num_repeats)
    record = {
        "task_id": int(task_id + 1),
        "task_description": task_description,
        "episode_idx": int(episode_idx + 1),
        "num_repeats": int(num_repeats),
        "success_count": success_count,
        "consistency_score": consistency_score,
        "repeat_successes": list(repeat_successes),
    }
    self_eval_records.append(record)
    write_self_eval_json(
        {
            "self_eval_mode": cfg.self_eval_mode,
            "libero_benchmark": getattr(cfg, "libero_benchmark", "libero"),
            "task_suite_name": cfg.task_suite_name,
            "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
            "consistency_repeats_per_init": int(cfg.consistency_repeats_per_init),
            "action_noise_std": float(getattr(cfg, "action_noise_std", 0.0)),
            "records": self_eval_records,
        },
        self_eval_json_path,
    )
    return consistency_score
