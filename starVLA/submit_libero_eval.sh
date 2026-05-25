#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: normal mode runs 50 trials; eval mode runs 3 trials.
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_NUM_TRIALS="${EVAL_NUM_TRIALS:-3}"
# 1: enable online visual MAE-C eval mode; 0: keep normal mode (save).
# Default is OFF.
ENABLE_ATTN_EVAL="${ENABLE_ATTN_EVAL:-0}"
# Strict same-initial-scene repeats and optional action noise.
CONSISTENCY_REPEATS_PER_INIT="${CONSISTENCY_REPEATS_PER_INIT:-1}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.0}"

for suite in libero_spatial libero_object libero_10 libero_goal; do
  echo "Submitting ${suite}..."
  EXTRA_ARGS=()
  if [[ "${ENABLE_ATTN_EVAL}" == "1" ]]; then
    EXTRA_ARGS+=(--num-trials "${EVAL_NUM_TRIALS}")
    EXTRA_ARGS+=(--attention-eval-mode eval --attention-eval-method mae-c --attention-eval-ratios 0.01)
  else
    EXTRA_ARGS+=(--num-trials "${NUM_TRIALS}")
  fi
  EXTRA_ARGS+=(--consistency-repeats-per-init "${CONSISTENCY_REPEATS_PER_INIT}")
  EXTRA_ARGS+=(--action-noise-std "${ACTION_NOISE_STD}")
  python "${SCRIPT_DIR}/submit_libero_eval.py" --suite "${suite}" --slurm --model pi "${EXTRA_ARGS[@]}"
  sleep 2
done