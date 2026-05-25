"""Path helpers for standalone starVLA under main/."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_MAIN_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def repo_root() -> Path:
    return _REPO_ROOT


def main_root() -> Path:
    return _MAIN_ROOT


def libero_home() -> Path:
    return Path(os.environ.get("LIBERO_HOME", str(_MAIN_ROOT / "third_party" / "LIBERO")))


def libero_pro_home() -> Path:
    return Path(os.environ.get("LIBERO_PRO_HOME", str(_MAIN_ROOT / "third_party" / "LIBERO_PRO")))


def libero_pro_eval_config() -> Path:
    env_path = os.environ.get("LIBERO_PRO_EVAL_CONFIG", "").strip()
    if env_path:
        return Path(env_path)
    return libero_pro_home() / "evaluation_config_swap.yaml"


def cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", str(_MAIN_ROOT.parent / ".cache")))


def eval_results_root() -> Path:
    return Path(os.environ.get("EVAL_RESULTS_ROOT", str(_REPO_ROOT / "eval_results")))


def save_attn_dir() -> Path:
    return Path(os.environ.get("SAVE_ATTN_DIR", str(_REPO_ROOT / "saved_attentions" / "Pi")))


def default_checkpoint() -> str:
    return os.environ.get("PRETRAINED_CHECKPOINT", "")
