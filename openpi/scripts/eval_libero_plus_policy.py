"""Evaluate an openpi websocket policy on LIBERO-plus tasks."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import pathlib
import re
import sys
import time
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import torch

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


PERTURBATION_ALIASES = {
    "all": None,
    "background": "Background Textures",
    "camera": "Camera Viewpoints",
    "language": "Language Instructions",
    "layout": "Objects Layout",
    "light": "Light Conditions",
    "noise": "Sensor Noise",
    "robot": "Robot Initial States",
}


def _default_libero_root() -> str:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    project_root = repo_root.parents[1]
    for candidate in project_root.glob("*_link"):
        libero_root = candidate / "libero" / "LIBERO-plus"
        if libero_root.exists():
            return str(libero_root)
    return str(repo_root / "LIBERO-plus")


def _select_task_ids(
    benchmark,
    libero_root: pathlib.Path,
    suite: str,
    perturbation: str,
) -> list[int]:
    category = PERTURBATION_ALIASES[perturbation]
    if category is None:
        return list(range(benchmark.get_num_tasks()))

    classification_path = libero_root / "libero" / "libero" / "benchmark" / "task_classification.json"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    if suite not in classification:
        supported = ", ".join(sorted(classification))
        raise ValueError(
            f"Perturbation filtering is unavailable for suite {suite!r}; "
            f"use one of the officially classified suites: {supported}"
        )

    names = {item["name"] for item in classification[suite] if item["category"] == category}
    task_ids = [task_id for task_id in range(benchmark.get_num_tasks()) if benchmark.get_task(task_id).name in names]
    if len(task_ids) != len(names):
        benchmark_names = {benchmark.get_task(task_id).name for task_id in range(benchmark.get_num_tasks())}
        missing = sorted(names - benchmark_names)
        raise RuntimeError(
            f"Classification mismatch for {suite}/{category}: selected {len(task_ids)} of {len(names)} tasks; "
            f"first missing tasks: {missing[:5]}"
        )
    return task_ids


def _as_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255)
        image = image.astype(np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def _first_obs(obs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"None of the observation keys exist: {keys}; available keys: {sorted(obs.keys())}")


def _state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs and "robot0_gripper_qpos" in obs:
        return np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"]).reshape(-1),
                _quat2axisangle(np.asarray(obs["robot0_eef_quat"]).copy()),
                np.asarray(obs["robot0_gripper_qpos"]).reshape(-1),
            ]
        ).astype(np.float32)

    for key in ("robot0_eef_state", "robot0_proprio-state", "robot0_joint_pos"):
        if key in obs:
            state = np.asarray(obs[key]).reshape(-1).astype(np.float32)
            if state.size >= 8:
                return state[:8]

    raise KeyError(f"Could not build an 8-dim LIBERO state from observation keys: {sorted(obs.keys())}")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _prepare_image(image: np.ndarray, resize_size: int) -> np.ndarray:
    # LIBERO training data is rotated 180 degrees by the official openpi evaluator.
    image = np.ascontiguousarray(_as_uint8_hwc(image)[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))


def _policy_obs(obs: dict[str, Any], prompt: str, resize_size: int) -> dict[str, Any]:
    base_image = _first_obs(obs, ("agentview_image", "frontview_image", "base_0_rgb"))
    wrist_image = _first_obs(
        obs, ("robot0_eye_in_hand_image", "eye_in_hand_image", "left_wrist_0_rgb", "agentview_image")
    )
    return {
        "observation/state": _state_from_obs(obs),
        "observation/image": _prepare_image(base_image, resize_size),
        "observation/wrist_image": _prepare_image(wrist_image, resize_size),
        "prompt": prompt,
    }


def _write_libero_config(libero_root: pathlib.Path, config_dir: pathlib.Path) -> pathlib.Path:
    benchmark_root = libero_root / "libero" / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_text = "\n".join(
        [
            f"assets: {benchmark_root / 'assets'}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"benchmark_root: {benchmark_root}",
            f"datasets: {libero_root / 'datasets'}",
            f"init_states: {benchmark_root / 'init_files'}",
            "",
        ]
    )
    temp_path = config_path.with_suffix(f".tmp.{os.getpid()}")
    temp_path.write_text(config_text, encoding="utf-8")
    temp_path.replace(config_path)
    return config_dir


def _resolve_init_states(benchmark, task_id: int) -> Any:
    task = benchmark.get_task(task_id)
    from libero.libero import get_libero_path

    init_file = task.init_states_file
    init_root = get_libero_path("init_states")
    init_states_path: str

    if "_add_" in init_file or re.search(r"_level\d+_sample\d+\.", init_file):
        init_states_path = os.path.join(init_root, "libero_newobj", task.problem_folder, init_file)
    else:
        stem, extension = init_file.rsplit(".", 1)
        # Match generated suffixes only at the end so words such as
        # "from_table_center" in the base task name are not misclassified.
        base_stem = re.sub(r"_language_\d+(?:_view_.*)?$", "", stem)
        base_stem = re.sub(r"_view_.*$", "", base_stem)
        base_stem = re.sub(r"_(?:table|tb|light)_\d+$", "", base_stem)
        init_states_path = os.path.join(init_root, task.problem_folder, f"{base_stem}.{extension}")

    init_states = torch.load(init_states_path, weights_only=False)
    if "_add_" in init_file or "_level" in init_file:
        init_states = init_states.reshape(1, -1)
    return init_states


def _resolve_language_bddl_path(bddl_file_name: str) -> pathlib.Path:
    path = pathlib.Path(bddl_file_name)
    if path.exists():
        return path

    # Camera, robot-state, and sensor-noise variants encode runtime parameters
    # after "_view_" in a virtual filename. The environment understands that
    # filename, but the language instruction lives in the physical base BDDL.
    if "_view_" in path.stem:
        base_path = path.with_name(path.stem.split("_view_", 1)[0] + path.suffix)
        if base_path.exists():
            return base_path

    raise FileNotFoundError(f"Could not resolve a physical BDDL file for {bddl_file_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", default=_default_libero_root())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", default="libero_mix")
    parser.add_argument("--perturbation", choices=sorted(PERTURBATION_ALIASES), default="all")
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--attention-eval-mode",
        choices=("off", "mac", "mad", "both"),
        default="off",
        help="Compute final-ODE-step visual attention metrics returned by the policy server.",
    )
    parser.add_argument(
        "--attention-eval-ratios",
        default="0.01,0.05,0.1,0.5",
        help="Comma-separated head ratios requested from the policy server.",
    )
    parser.add_argument("--tts-mode", choices=("none", "independent", "branch"), default="none")
    parser.add_argument("--tts-num-candidates", type=int, default=4)
    parser.add_argument("--tts-selection-ratio", type=float, default=0.10)
    parser.add_argument("--tts-branch-ratio", type=float, default=0.40)
    parser.add_argument("--tts-branch-noise-scale", type=float, default=0.10)
    parser.add_argument(
        "--tts-score-mode",
        choices=("mae", "velocity_diff", "mae_diff", "mae_velocity_diff"),
        default="mae",
    )
    parser.add_argument("--flow-mg-mask", choices=("language", "vision", "language_vision"), default="language")
    parser.add_argument("--flow-mg-steps", default="4,7,9")
    parser.add_argument(
        "--save-attention-metrics",
        action="store_true",
        help="Write per-policy-query attention metrics to attention_metrics.jsonl. Disabled by default.",
    )
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attention_eval_ratios = tuple(float(item.strip()) for item in args.attention_eval_ratios.split(",") if item.strip())
    if not attention_eval_ratios or any(ratio <= 0 or ratio > 1 for ratio in attention_eval_ratios):
        raise ValueError(f"Attention eval ratios must be in (0, 1], got: {attention_eval_ratios}")
    if args.tts_mode == "none":
        attention_scalar_metrics = tuple(
            f"{metric}_{selection}{round(ratio * 100)}"
            for metric, selection in (("mac", "bottom"), ("mad", "top"))
            for ratio in attention_eval_ratios
            if args.attention_eval_mode in ("both", metric)
        )
    else:
        attention_scalar_metrics = tuple(
            f"{metric}_{selection}{round(ratio * 100)}_selected"
            for metric, selection in (("mac", "bottom"), ("mad", "top"))
            for ratio in attention_eval_ratios
            if args.attention_eval_mode in ("both", metric)
        )
        attention_scalar_metrics = attention_scalar_metrics + tuple(
            f"mae_bottom{round(ratio * 100)}_selected" for ratio in attention_eval_ratios
        )
        attention_scalar_metrics = (
            *attention_scalar_metrics,
            "tts_selection_score",
            "tts_best_candidate_index",
            "tts_score_mode_velocity_diff",
            "tts_score_mode_mae_diff",
            "tts_score_mode_mae_velocity_diff",
            "flow_mg_velocity_diff_selected",
            "flow_mg_mae_diff_selected",
            "flow_mg_mae_velocity_diff_selected",
            "flow_mg_num_scored_steps",
        )
    eval_start = time.monotonic()
    libero_root = pathlib.Path(args.libero_root).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(libero_root))
    os.environ["LIBERO_CONFIG_PATH"] = str(_write_libero_config(libero_root, args.out_dir / ".libero_config"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")

    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import get_problem_info

    policy = WebsocketClientPolicy(args.host, args.port)
    benchmark = get_benchmark(args.suite)(args.task_order_index)

    category_task_ids = _select_task_ids(benchmark, libero_root, args.suite, args.perturbation)
    end_task_offset = min(args.task_start + args.num_tasks, len(category_task_ids))
    selected_task_ids = category_task_ids[args.task_start : end_task_offset]
    total_tasks = len(selected_task_ids)
    total_episodes = total_tasks * args.episodes_per_task
    rows: list[dict[str, Any]] = []
    attention_path = args.out_dir / "attention_metrics.jsonl"
    attention_file = (
        attention_path.open("w", encoding="utf-8")
        if args.attention_eval_mode != "off" and args.save_attention_metrics
        else None
    )
    attention_metric_sums: dict[str, float] = collections.defaultdict(float)
    attention_metric_counts: dict[str, int] = collections.defaultdict(int)

    print(
        json.dumps(
            {
                "event": "evaluation_start",
                "suite": args.suite,
                "perturbation": args.perturbation,
                "category": PERTURBATION_ALIASES[args.perturbation],
                "category_num_tasks": len(category_task_ids),
                "task_offset_start": args.task_start,
                "task_offset_end": end_task_offset - 1,
                "num_tasks": total_tasks,
                "episodes_per_task": args.episodes_per_task,
                "total_episodes": total_episodes,
                "max_steps": args.max_steps,
                "camera_resolution": [args.camera_height, args.camera_width],
                "resize_size": args.resize_size,
                "replan_steps": args.replan_steps,
                "num_steps_wait": args.num_steps_wait,
                "seed": args.seed,
                "attention_eval_mode": args.attention_eval_mode,
                "attention_eval_ratios": attention_eval_ratios,
                "save_attention_metrics": args.save_attention_metrics,
                "tts_mode": args.tts_mode,
                "tts_num_candidates": args.tts_num_candidates,
                "tts_selection_ratio": args.tts_selection_ratio,
                "tts_branch_ratio": args.tts_branch_ratio,
                "tts_branch_noise_scale": args.tts_branch_noise_scale,
                "tts_score_mode": args.tts_score_mode,
                "flow_mg_mask": args.flow_mg_mask,
                "flow_mg_steps": args.flow_mg_steps,
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for task_number, task_id in enumerate(selected_task_ids, start=1):
        task = benchmark.get_task(task_id)
        init_states = _resolve_init_states(benchmark, task_id)
        init_states = np.asarray(init_states)
        bddl_file_name = benchmark.get_task_bddl_file_path(task_id)
        language_bddl_path = _resolve_language_bddl_path(bddl_file_name)
        language = get_problem_info(str(language_bddl_path))["language_instruction"]
        env_args = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": args.camera_height,
            "camera_widths": args.camera_width,
        }
        successes = []
        task_start = time.monotonic()

        print(
            json.dumps(
                {
                    "event": "task_start",
                    "suite": args.suite,
                    "perturbation": args.perturbation,
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_number": task_number,
                    "total_tasks": total_tasks,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        for episode in range(args.episodes_per_task):
            episode_start = time.monotonic()
            env = OffScreenRenderEnv(**env_args)
            env.seed(args.seed)
            env.reset()
            init_idx = episode % len(init_states)
            obs = env.set_init_state(init_states[init_idx])
            policy.reset()
            action_plan: collections.deque[np.ndarray] = collections.deque()
            episode_attention_metrics: dict[str, list[float]] = collections.defaultdict(list)
            policy_query = 0

            success = False
            steps = 0
            episode_error = ""
            try:
                for _ in range(args.num_steps_wait):
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    if bool(done):
                        success = True
                        break

                if not success:
                    for steps in range(1, args.max_steps + 1):
                        if not action_plan:
                            response = policy.infer(_policy_obs(obs, language, args.resize_size))
                            action_chunk = np.asarray(response["actions"])
                            policy_query += 1
                            if args.attention_eval_mode != "off":
                                metrics = response.get("attention_metrics")
                                if metrics is None:
                                    raise RuntimeError(
                                        "Attention eval was requested, but the policy response has no attention_metrics. "
                                        "Start the policy server with --attention-eval."
                                    )
                                selected_metrics = {}
                                for name, value in metrics.items():
                                    is_tts_metric = name.startswith(("mae_", "tts_", "flow_mg_"))
                                    if args.attention_eval_mode == "mac" and not (
                                        name.startswith("mac_") or is_tts_metric
                                    ):
                                        continue
                                    if args.attention_eval_mode == "mad" and not (
                                        name.startswith("mad_") or is_tts_metric
                                    ):
                                        continue
                                    array = np.asarray(value)
                                    selected_metrics[name] = array.tolist() if array.ndim else float(array)
                                    if not name.endswith(("_layers", "_steps_layers")) and array.ndim == 0:
                                        episode_attention_metrics[name].append(float(array))
                                if attention_file is not None:
                                    record = {
                                        "suite": args.suite,
                                        "perturbation": args.perturbation,
                                        "task_id": task_id,
                                        "task_name": task.name,
                                        "episode": episode,
                                        "policy_query": policy_query,
                                        "environment_step": steps,
                                        "metrics": selected_metrics,
                                    }
                                    attention_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                                for name, value in selected_metrics.items():
                                    if not name.endswith(("_layers", "_steps_layers")) and np.asarray(value).ndim == 0:
                                        attention_metric_sums[name] += float(value)
                                        attention_metric_counts[name] += 1
                            if action_chunk.ndim == 1:
                                action_chunk = action_chunk[None, :]
                            if len(action_chunk) < args.replan_steps:
                                raise ValueError(
                                    f"Policy returned {len(action_chunk)} actions, but replan_steps={args.replan_steps}"
                                )
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = np.asarray(action_plan.popleft())
                        obs, _, done, _ = env.step(action[:7])
                        if bool(done):
                            success = True
                            break
            except Exception as exc:
                episode_error = f"{type(exc).__name__}: {exc}"
                print(
                    json.dumps(
                        {
                            "event": "episode_error",
                            "suite": args.suite,
                            "perturbation": args.perturbation,
                            "task_id": task_id,
                            "task_name": task.name,
                            "episode": episode,
                            "steps": steps,
                            "error": episode_error,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            finally:
                env.close()

            successes.append(float(success))
            now = time.monotonic()
            completed_episodes = len(rows) + 1
            total_elapsed_sec = now - eval_start
            eta_sec = (
                total_elapsed_sec / completed_episodes * (total_episodes - completed_episodes)
                if completed_episodes and total_episodes
                else 0.0
            )
            row = {
                "event": "episode_complete",
                "suite": args.suite,
                "perturbation": args.perturbation,
                "task_id": task_id,
                "task_name": task.name,
                "language": language,
                "episode": episode,
                "success": int(success),
                "steps": steps,
                "episode_error": episode_error,
                "episode_elapsed_sec": round(now - episode_start, 3),
                "task_elapsed_sec": round(now - task_start, 3),
                "total_elapsed_sec": round(total_elapsed_sec, 3),
                "completed_episodes": completed_episodes,
                "total_episodes": total_episodes,
                "eta_sec": round(eta_sec, 3),
                **{
                    name: (
                        float(np.mean(episode_attention_metrics[name]))
                        if episode_attention_metrics[name]
                        else float("nan")
                    )
                    for name in attention_scalar_metrics
                },
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        completed_tasks = task_number
        total_elapsed_sec = time.monotonic() - eval_start
        task_eta_sec = total_elapsed_sec / completed_tasks * (total_tasks - completed_tasks) if completed_tasks else 0.0
        print(
            json.dumps(
                {
                    "event": "task_complete",
                    "suite": args.suite,
                    "perturbation": args.perturbation,
                    "task_id": task_id,
                    "task_name": task.name,
                    "success_rate": float(np.mean(successes)),
                    "episodes": len(successes),
                    "task_elapsed_sec": round(time.monotonic() - task_start, 3),
                    "total_elapsed_sec": round(total_elapsed_sec, 3),
                    "completed_tasks": completed_tasks,
                    "total_tasks": total_tasks,
                    "eta_sec": round(task_eta_sec, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    csv_path = args.out_dir / "episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["suite", "task_id", "success"])
        writer.writeheader()
        writer.writerows(rows)

    if attention_file is not None:
        attention_file.close()
    attention_summary = {
        name: attention_metric_sums[name] / attention_metric_counts[name]
        for name in sorted(attention_metric_counts)
        if attention_metric_counts[name]
    }

    summary = {
        "event": "evaluation_complete",
        "suite": args.suite,
        "perturbation": args.perturbation,
        "category": PERTURBATION_ALIASES[args.perturbation],
        "task_start": args.task_start,
        "num_tasks": total_tasks,
        "episodes": len(rows),
        "successful_episodes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "elapsed_sec": round(time.monotonic() - eval_start, 3),
        "episodes_csv": str(csv_path),
        "attention_eval_mode": args.attention_eval_mode,
        "attention_eval_ratios": attention_eval_ratios,
        "save_attention_metrics": args.save_attention_metrics,
        "tts_mode": args.tts_mode,
        "tts_num_candidates": args.tts_num_candidates,
        "tts_selection_ratio": args.tts_selection_ratio,
        "tts_branch_ratio": args.tts_branch_ratio,
        "tts_branch_noise_scale": args.tts_branch_noise_scale,
        "tts_score_mode": args.tts_score_mode,
        "flow_mg_mask": args.flow_mg_mask,
        "flow_mg_steps": args.flow_mg_steps,
        "attention_metrics_jsonl": str(attention_path) if attention_file is not None else None,
        "attention_metrics": attention_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
