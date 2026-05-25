"""Path helpers for standalone openvla under main/."""
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


def libero_path() -> Path:
    return Path(os.environ.get("LIBERO_PATH", str(_MAIN_ROOT / "third_party" / "LIBERO")))


def libero_pro_root() -> Path:
    return Path(os.environ.get("LIBERO_PRO_PATH", str(_MAIN_ROOT / "third_party" / "LIBERO_PRO")))


def libero_pro_code_root() -> Path:
    return Path(
        os.environ.get(
            "LIBERO_PRO_CODE_ROOT",
            str(_MAIN_ROOT / "openvla_oft" / "openvla-oft_code"),
        )
    )


def cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", str(_MAIN_ROOT.parent / ".cache")))


def eval_results_root() -> Path:
    return Path(os.environ.get("OPENVLA_EVAL_RESULTS_ROOT", str(_REPO_ROOT / "eval_results")))
