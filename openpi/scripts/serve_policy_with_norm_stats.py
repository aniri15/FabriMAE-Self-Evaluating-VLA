"""Serve an openpi checkpoint with optional normalization stats override.

This is useful for evaluating pi05_base on LIBERO: the base checkpoint does not
ship LIBERO normalization stats, so we load those from pi05_libero while keeping
the weights from the requested checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import socket

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


def _parse_ratios(value: str) -> tuple[float, ...]:
    ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not ratios or any(ratio <= 0 or ratio > 1 for ratio in ratios):
        raise argparse.ArgumentTypeError("Ratios must be a comma-separated list of values in (0, 1].")
    return ratios


def _parse_flow_mg_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("Flow-MG steps must be a comma-separated list of non-negative integers.")
    return steps


def _attention_method_to_mode(method: str) -> str:
    normalized = method.lower().replace("_", "-")
    if normalized in ("mae-c", "mac"):
        return "mac"
    if normalized in ("mae-d", "mad"):
        return "mad"
    if normalized == "both":
        return "both"
    raise argparse.ArgumentTypeError("Attention eval method must be mae-c, mae-d, mac, mad, or both.")


def _load_norm_stats(config_name: str, checkpoint_dir: pathlib.Path, norm_stats_checkpoint: pathlib.Path | None):
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError(f"Config {config_name!r} does not define a data asset id.")

    candidates = []
    if norm_stats_checkpoint is not None:
        candidates.append(norm_stats_checkpoint)
    candidates.append(checkpoint_dir)

    errors = []
    for candidate in candidates:
        try:
            return _checkpoints.load_norm_stats(candidate / "assets", data_config.asset_id)
        except FileNotFoundError as exc:
            errors.append(f"{candidate / 'assets' / data_config.asset_id}: {exc}")

    raise FileNotFoundError("Could not load norm stats from any checkpoint:\n" + "\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_libero", help="openpi training config name")
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--norm-stats-checkpoint",
        type=pathlib.Path,
        default=None,
        help="Checkpoint whose assets/<asset_id>/norm_stats.json should be used.",
    )
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--attention-eval", action="store_true", help="Return final-ODE-step MAC/MAD metrics.")
    parser.add_argument("--attention-eval-mode", choices=("mac", "mad", "both"), default="both")
    parser.add_argument(
        "--attention-eval-method",
        type=_attention_method_to_mode,
        default=None,
        help="User-facing attention method. mae-c maps to mac; mae-d maps to mad.",
    )
    parser.add_argument(
        "--attention-eval-ratios",
        type=_parse_ratios,
        default=(0.01, 0.05, 0.10, 0.50),
        help="Comma-separated head ratios, for example 0.01 or 0.01,0.05,0.1,0.5.",
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
    parser.add_argument(
        "--flow-mg-steps",
        type=_parse_flow_mg_steps,
        default=(4, 7, 9),
        help="Comma-separated ODE step indices used by velocity_diff scoring, for example 4,7,9.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.attention_eval_method is not None:
        args.attention_eval_mode = args.attention_eval_method
    train_config = _config.get_config(args.config)
    norm_stats = _load_norm_stats(args.config, args.checkpoint_dir, args.norm_stats_checkpoint)

    logging.info("Loading policy config=%s checkpoint=%s", args.config, args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
        attention_eval=args.attention_eval,
        attention_eval_mode=args.attention_eval_mode,
        attention_eval_ratios=args.attention_eval_ratios,
        tts_mode=args.tts_mode,
        tts_num_candidates=args.tts_num_candidates,
        tts_selection_ratio=args.tts_selection_ratio,
        tts_branch_ratio=args.tts_branch_ratio,
        tts_branch_noise_scale=args.tts_branch_noise_scale,
        tts_score_mode=args.tts_score_mode,
        flow_mg_mask=args.flow_mg_mask,
        flow_mg_steps=args.flow_mg_steps,
    )
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    logging.info("Serving policy on %s:%s from host %s", args.host, args.port, hostname)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
