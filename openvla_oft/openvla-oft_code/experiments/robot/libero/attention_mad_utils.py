"""Online MAE-D attention scoring for OpenVLA / OpenVLA-OFT LIBERO rollouts (internal compute: mad)."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from experiments.robot.libero.mae_metric_naming import (
    MAE_D_MODE,
    external_metric_label,
    is_mae_d_self_eval_mode,
    mae_d_score_key,
)


def parse_attention_eval_ratios(ratios_str: str) -> List[float]:
    ratios: List[float] = []
    for part in str(ratios_str).split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if val <= 0 or val > 1:
            raise ValueError(f"Invalid ratio: {val}")
        ratios.append(val)
    if not ratios:
        raise ValueError("No valid ratios parsed")
    return ratios


def _batch_entropy(probs: np.ndarray, axis: int = -1, epsilon: float = 1e-12) -> np.ndarray:
    row_sums = np.sum(probs, axis=axis, keepdims=True)
    probs_norm = probs / (row_sums + epsilon)
    probs_norm = np.clip(probs_norm, epsilon, 1.0)
    return -np.sum(probs_norm * np.log(probs_norm), axis=axis)


def _stack_trim_mats(mats: Sequence[np.ndarray]) -> np.ndarray:
    min_l = min(m.shape[0] for m in mats)
    min_h = min(m.shape[1] for m in mats)
    trimmed = [m[:min_l, :min_h] for m in mats]
    return np.stack(trimmed, axis=0)


def compute_layer_k_entropy_scores_for_ratios(
    entropy_lh: np.ndarray,
    ratios: Sequence[float],
    *,
    select_bottom: bool,
    negate_mean: bool,
) -> dict:
    """Layer-wise top/bottom-k head entropy means (MAD uses top-k + negate_mean)."""
    arr = np.nan_to_num(np.asarray(entropy_lh, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2:
        raise ValueError(f"Expected (L, H) entropy matrix, got shape: {arr.shape}")
    num_layers, num_heads = arr.shape
    if num_layers == 0 or num_heads == 0:
        return {ratio: np.zeros((0,), dtype=np.float64) for ratio in ratios}

    sorted_rows = np.sort(arr, axis=1)
    cumsum_rows = np.cumsum(sorted_rows, axis=1)
    total_rows = cumsum_rows[:, -1]

    ratio_scores = {}
    for ratio in ratios:
        k = max(1, int(np.ceil(num_heads * ratio)))
        if select_bottom:
            vals = cumsum_rows[:, k - 1] / k
        else:
            vals = (
                (total_rows - cumsum_rows[:, num_heads - k - 1]) / k
                if k < num_heads
                else total_rows / k
            )
        if negate_mean:
            vals = -vals
        ratio_scores[ratio] = vals
    return ratio_scores


def _openvla_layers_from_attentions(
    saved_attentions: Any,
    num_patches: int,
    num_prompt_tokens: int,
) -> Optional[List[np.ndarray]]:
    if saved_attentions is None:
        return None

    text_end = num_patches + num_prompt_tokens
    chunks = saved_attentions[1:] if isinstance(saved_attentions, tuple) and len(saved_attentions) > 1 else (saved_attentions,)

    step_arrays = []
    for saved_attention in chunks:
        if saved_attention is None:
            continue
        layer_arrays = []
        for layer_attention in saved_attention:
            tensor = layer_attention[:, :, :, :text_end]
            if hasattr(tensor, "float"):
                tensor = tensor.float().cpu().numpy()
            else:
                tensor = np.asarray(tensor)
            layer_arrays.append(tensor)
        if layer_arrays:
            step_arrays.append(np.array(layer_arrays))

    if not step_arrays:
        return None

    stacked = np.concatenate(step_arrays, axis=3).swapaxes(0, 1)[0]
    return [stacked[l] for l in range(stacked.shape[0])]


def _openvla_oft_layers_from_attentions(attentions: Any) -> Optional[List[np.ndarray]]:
    if attentions is None or not isinstance(attentions, tuple):
        return None
    layers = []
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        if hasattr(layer_attn, "float"):
            layer_attn = layer_attn.float().cpu().numpy()
        else:
            layer_attn = np.asarray(layer_attn, dtype=np.float32)
        layers.append(layer_attn[0])
    return layers or None


def compute_visual_entropy_lh_from_attentions(
    attentions: Any,
    num_patches: int,
    num_prompt_tokens: int,
    *,
    variant: str = "openvla",
) -> Optional[np.ndarray]:
    """
    Extract per-layer, per-head visual entropy (L, H) for one policy query.

    MAD: entropy over visual tokens, mean over action rows; top-k head mean with negated sign.
    """
    if num_patches is None or num_prompt_tokens is None:
        return None

    variant = str(variant).strip().lower()
    layers: Optional[List[np.ndarray]] = None

    if variant == "openvla_oft":
        layers = _openvla_oft_layers_from_attentions(attentions)
        if layers is None:
            return None
        num_actions_chunk = 8
        action_dim = 7
        num_action_tokens = num_actions_chunk * action_dim
        visual_start, visual_end = 0, num_patches
        text_start = num_patches
        text_end = num_patches + num_prompt_tokens
        action_start = text_end
        action_end = action_start + num_action_tokens
        head_entropies_per_layer = []
        for layer_attn in layers:
            num_heads = layer_attn.shape[0]
            head_vals = []
            for h in range(num_heads):
                head_block = layer_attn[h, action_start:action_end, :]
                vis_ent = _batch_entropy(head_block[:, visual_start:visual_end])
                head_vals.append(float(np.mean(vis_ent)))
            head_entropies_per_layer.append(np.asarray(head_vals, dtype=np.float32))
        return np.stack(head_entropies_per_layer, axis=0)

    if variant != "openvla":
        raise ValueError(f"Unsupported attention variant: {variant}")

    layers = _openvla_layers_from_attentions(attentions, num_patches, num_prompt_tokens)
    if layers is None:
        return None

    visual_start, visual_end = 1, num_patches + 1
    head_entropies_per_layer = []
    for layer_attn in layers:
        num_heads = layer_attn.shape[0]
        head_vals = []
        for h in range(num_heads):
            head_block = layer_attn[h, :, :]
            vis_ent = _batch_entropy(head_block[:, visual_start:visual_end])
            head_vals.append(float(np.mean(vis_ent)))
        head_entropies_per_layer.append(np.asarray(head_vals, dtype=np.float32))
    return np.stack(head_entropies_per_layer, axis=0)


def build_episode_mad_record(
    *,
    task_id: int,
    task_description: str,
    episode_idx: int,
    repeat_idx: int,
    episode_number: int,
    is_success: bool,
    episode_visual_entropy_queries: Sequence[np.ndarray],
    eval_ratios: Sequence[float],
) -> Optional[dict]:
    if not episode_visual_entropy_queries:
        return None
    episode_entropy_lh = np.nanmean(_stack_trim_mats(episode_visual_entropy_queries), axis=0)
    layer_scores_by_ratio = compute_layer_k_entropy_scores_for_ratios(
        episode_entropy_lh,
        ratios=eval_ratios,
        select_bottom=False,
        negate_mean=True,
    )
    metric_scores = {
        mae_d_score_key(ratio): float(np.mean(layer_scores_by_ratio[ratio]))
        for ratio in eval_ratios
    }
    return {
        "task_id": int(task_id),
        "task_description": task_description,
        "episode_idx": int(episode_idx),
        "repeat_idx": int(repeat_idx),
        "episode_number": int(episode_number),
        "is_success": bool(is_success),
        "num_queries_used": int(len(episode_visual_entropy_queries)),
        "attention_eval_method": external_metric_label(MAE_D_MODE),
        "scores": metric_scores,
    }


def append_online_mad_jsonl(record: dict, jsonl_path: str) -> None:
    if not jsonl_path or record is None:
        return
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def rollout_requests_attentions(cfg) -> bool:
    if getattr(cfg, "save_attentions", False):
        return True
    return bool(
        getattr(cfg, "enable_self_eval", False)
        and is_mae_d_self_eval_mode(getattr(cfg, "self_eval_mode", ""))
    )
