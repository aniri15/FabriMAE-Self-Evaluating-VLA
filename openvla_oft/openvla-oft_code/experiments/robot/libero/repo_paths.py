"""Relative paths for OFT LIBERO eval scripts."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_LIBERO_DIR = Path(__file__).resolve().parent
_OFT_ROOT = _LIBERO_DIR.parents[2]
_REPO_ROOT = _OFT_ROOT.parent
_MAIN_ROOT = _REPO_ROOT.parent

if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from _shared_paths import (  # noqa: E402
    libero_pro_config_dir,
    libero_pro_evaluation_config_swap,
    libero_pro_pkg_root,
    libero_standard_config_dir,
    standard_libero_pkg_root,
    third_party_libero,
    third_party_libero_pro,
)


def bundled_libero_root() -> Path:
    return standard_libero_pkg_root()


def libero_pro_libero_root() -> Path:
    return libero_pro_pkg_root()


def libero_pro_root() -> Path:
    return third_party_libero_pro()


def libero_home() -> Path:
    return third_party_libero()


def setup_libero_config(libero_pkg_root: Path, config_dir: Path | None = None) -> Path:
    if config_dir is None:
        if libero_pkg_root == libero_pro_pkg_root().resolve():
            config_dir = libero_pro_config_dir()
        else:
            config_dir = libero_standard_config_dir()
    else:
        config_dir = Path(config_dir)

    config_dir.mkdir(parents=True, exist_ok=True)
    root = libero_pkg_root.resolve()
    config_data = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root.parent.parent / "datasets"),
        "assets": str(root / "assets"),
    }
    config_file = config_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as handle:
        yaml.dump(config_data, handle)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return config_dir


def default_evaluation_config_swap() -> Path:
    env_path = os.environ.get("LIBERO_PRO_EVAL_CONFIG", "").strip()
    if env_path:
        return Path(env_path)
    return libero_pro_evaluation_config_swap()
