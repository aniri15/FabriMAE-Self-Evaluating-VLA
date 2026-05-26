"""Path helpers for standalone openvla_oft under main/."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_MAIN_ROOT = _REPO_ROOT.parent
_OFT_CODE = _REPO_ROOT / "openvla-oft_code"

if str(_OFT_CODE) not in sys.path:
    sys.path.insert(0, str(_OFT_CODE))


def repo_root() -> Path:
    return _REPO_ROOT


def main_root() -> Path:
    return _MAIN_ROOT


def oft_code_root() -> Path:
    return _OFT_CODE


def libero_path() -> Path:
    return Path(
        os.environ.get("LIBERO_PATH", str(_MAIN_ROOT / "third_party" / "LIBERO-REFLECT" / "standard"))
    ).expanduser().resolve()


def libero_pro_root() -> Path:
    return Path(
        os.environ.get("LIBERO_PRO_PATH", str(_MAIN_ROOT / "third_party" / "LIBERO-REFLECT" / "reflect"))
    ).expanduser().resolve()


def libero_pro_libero_root() -> Path:
    return libero_pro_root() / "libero" / "libero"


def cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", str(_MAIN_ROOT.parent / ".cache")))


def eval_results_root() -> Path:
    return Path(os.environ.get("OPENVLA_EVAL_RESULTS_ROOT", str(_REPO_ROOT / "eval_results")))
