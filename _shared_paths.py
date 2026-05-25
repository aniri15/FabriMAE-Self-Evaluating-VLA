"""Shared path helpers for standalone main/ subprojects."""
from __future__ import annotations

import os
from pathlib import Path


def main_root() -> Path:
    return Path(__file__).resolve().parent


def third_party_libero() -> Path:
    return main_root() / "third_party" / "LIBERO"


def third_party_libero_pro() -> Path:
    return main_root() / "third_party" / "LIBERO_PRO"


def standard_libero_pkg_root() -> Path:
    """Python package root: .../LIBERO/libero/libero"""
    return third_party_libero() / "libero" / "libero"


def libero_pro_pkg_root() -> Path:
    """Python package root: .../LIBERO_PRO/libero/libero"""
    return third_party_libero_pro() / "libero" / "libero"


def libero_standard_config_dir() -> Path:
    return third_party_libero() / ".libero_config"


def libero_pro_config_dir() -> Path:
    return third_party_libero_pro() / "libero_config"


def libero_pro_evaluation_config_swap() -> Path:
    return third_party_libero_pro() / "evaluation_config_swap.yaml"


def cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", str(main_root().parent / ".cache")))


def env_or_default(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default
