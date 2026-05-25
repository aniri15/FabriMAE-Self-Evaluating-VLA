import os
from typing import Iterable, List, Optional

import numpy as np


def compute_action_agreement(action_trajectories: Iterable[np.ndarray]) -> Optional[float]:
    """
    Mean pairwise action agreement across repeats for one fixed init scenario.
    Uses exp(-RMSE) over aligned timesteps so higher = more similar actions.
    """
    trajectories = [np.asarray(x, dtype=np.float64) for x in action_trajectories if x is not None]
    if len(trajectories) < 2:
        return None

    sims = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            a = trajectories[i]
            b = trajectories[j]
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
                continue
            min_len = min(len(a), len(b))
            if min_len <= 0:
                continue
            diff = a[:min_len] - b[:min_len]
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            sims.append(float(np.exp(-rmse)))
    if not sims:
        return None
    return float(np.mean(sims))


def save_repeat_action_array(action_dir: str, task_id: int, episode_idx: int, repeat_idx: int, actions: np.ndarray) -> str:
    os.makedirs(action_dir, exist_ok=True)
    rel_name = f"task_{int(task_id)}_episode_{int(episode_idx)}_repeat_{int(repeat_idx)}.npy"
    abs_path = os.path.join(action_dir, rel_name)
    np.save(abs_path, np.asarray(actions, dtype=np.float32))
    return abs_path


def build_action_consistency_record(
    *,
    task_id: int,
    task_description: str,
    episode_idx: int,
    repeat_idx: int,
    episode_number: int,
    is_success: bool,
    action_path: str,
    scenario_action_consistency: Optional[float],
    num_repeats_in_scenario: int,
) -> dict:
    return {
        "task_id": int(task_id),
        "task_description": str(task_description),
        "episode_idx": int(episode_idx),
        "repeat_idx": int(repeat_idx),
        "episode_number": int(episode_number),
        "is_success": bool(is_success),
        "action_path": str(action_path),
        "scenario_action_consistency": scenario_action_consistency,
        "num_repeats_in_scenario": int(num_repeats_in_scenario),
    }
