#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MAIN_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# shellcheck source=/dev/null
source "${MAIN_ROOT}/third_party/libero_env.sh"
setup_libero_env "${LIBERO_BENCHMARK:-libero}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export DATE_STAMP="${DATE_STAMP:-$(date +%Y%m%d)}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt="${PRETRAINED_CHECKPOINT:-${CHECKPOINT_PI:-}}"

if [[ -z "${your_ckpt}" ]]; then
  echo "Set PRETRAINED_CHECKPOINT or CHECKPOINT_PI to the model checkpoint path."
  exit 1
fi

cd "${REPO_ROOT}"
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

task_suite_name=libero_spatial
num_trials_per_task=50
video_out_path="results/${task_suite_name}"

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${your_ckpt}" \
  --args.host "${host}" \
  --args.port "${base_port}" \
  --args.task-suite-name "${task_suite_name}" \
  --args.num-trials-per-task "${num_trials_per_task}" \
  --args.video-out-path "${video_out_path}"
