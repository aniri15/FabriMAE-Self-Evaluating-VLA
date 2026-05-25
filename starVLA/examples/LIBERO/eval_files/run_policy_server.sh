#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MAIN_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
export HF_HOME="${HF_HOME:-${CACHE_ROOT:-${MAIN_ROOT}/../.cache}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

your_ckpt="${PRETRAINED_CHECKPOINT:-${CHECKPOINT_PI:-}}"
gpu_id="${GPU_ID:-0}"
port="${PORT:-5694}"
python_bin="${PYTHON:-python}"

if [[ -z "${your_ckpt}" ]]; then
  echo "Set PRETRAINED_CHECKPOINT or CHECKPOINT_PI."
  exit 1
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" deployment/model_server/server_policy.py \
  --ckpt_path "${your_ckpt}" \
  --port "${port}" \
  --use_bf16
