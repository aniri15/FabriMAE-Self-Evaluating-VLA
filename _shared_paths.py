"""Shared path helpers for repo-root subprojects."""
from __future__ import annotations

import os
from pathlib import Path


def main_root() -> Path:
    return Path(__file__).resolve().parent


def libero_reflect_root() -> Path:
    env = os.environ.get("LIBERO_REFLECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (main_root() / "third_party" / "LIBERO-REFLECT").resolve()


def third_party_libero() -> Path:
    env = os.environ.get("LIBERO_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (libero_reflect_root() / "standard").resolve()


def third_party_libero_pro() -> Path:
    """Reflect / LIBERO-PRO perturbation tree (legacy name: libero_pro)."""
    env = os.environ.get("LIBERO_PRO_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    env = os.environ.get("LIBERO_PRO_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (libero_reflect_root() / "reflect").resolve()


third_party_libero_reflect = third_party_libero_pro


def standard_libero_pkg_root() -> Path:
    return third_party_libero() / "libero" / "libero"


def libero_pro_pkg_root() -> Path:
    return third_party_libero_pro() / "libero" / "libero"


libero_reflect_pkg_root = libero_pro_pkg_root


def libero_standard_config_dir() -> Path:
    return third_party_libero() / ".libero_config"


def libero_pro_config_dir() -> Path:
    return third_party_libero_pro() / "libero_config"


def libero_pro_evaluation_config_swap() -> Path:
    env = os.environ.get("LIBERO_PRO_EVAL_CONFIG", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    framework_cfg = libero_reflect_root() / "model_configs" / "evaluation_config_swap.yaml"
    if framework_cfg.is_file():
        return framework_cfg
    return third_party_libero_pro() / "evaluation_config_swap.yaml"


def cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", str(main_root().parent / ".cache")))


def env_or_default(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default
