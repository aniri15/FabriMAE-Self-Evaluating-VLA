"""User-facing MAE-D / MAE-C names vs internal mad / mac computation."""

from __future__ import annotations

MAE_NAMING_NOTE = (
    "Metric naming: MAE-D (user-facing; internal compute: mad, top-k visual entropy) "
    "and MAE-C (user-facing; internal compute: mac, bottom-k visual entropy)."
)

MAE_D_MODE = "mae-d"
MAE_C_METHOD = "mae-c"
DEFAULT_MAE_D_OUTPUT = "online_MAE-D_scores.jsonl"
DEFAULT_MAE_C_OUTPUT = "online_MAE-C_scores.jsonl"

_MODE_ALIASES = {
    "mad": MAE_D_MODE,
    "mae_d": MAE_D_MODE,
    "mae-d": MAE_D_MODE,
}

_METHOD_ALIASES = {
    "mad": MAE_D_MODE,
    "mae_d": MAE_D_MODE,
    "mae-d": MAE_D_MODE,
    "mac": MAE_C_METHOD,
    "mae_c": MAE_C_METHOD,
    "mae-c": MAE_C_METHOD,
}


def _norm_token(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def normalize_self_eval_mode(mode: str) -> str:
    key = _norm_token(mode)
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    return key


def is_mae_d_self_eval_mode(mode: str) -> bool:
    return normalize_self_eval_mode(mode) == MAE_D_MODE


def normalize_attention_method(method: str) -> str:
    key = _norm_token(method)
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unsupported attention_eval_method={method!r}. Expected 'mae-d' or 'mae-c'."
        )
    return _METHOD_ALIASES[key]


def internal_compute_metric(method: str) -> str:
    """Return internal compute key: 'mad' or 'mac'."""
    normalized = normalize_attention_method(method)
    return "mad" if normalized == MAE_D_MODE else "mac"


def external_metric_label(method: str) -> str:
    return "MAE-D" if normalize_attention_method(method) == MAE_D_MODE else "MAE-C"


def mae_d_score_key(ratio: float) -> str:
    return f"MAE-D_top{int(ratio * 100)}"


def mae_c_score_key(ratio: float) -> str:
    return f"MAE-C_bottom{int(ratio * 100)}"
