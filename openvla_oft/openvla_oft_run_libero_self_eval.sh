#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --partition=a100

# Usage:
#   bash openvla_oft_run_libero_self_eval.sh [libero|libero_pro] [task_suite] [consistency|mae-d]
# MAE-D online: SELF_EVAL_MODE=mae-d (writes online_MAE-D_scores.jsonl; no attention .pt save)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
MAIN_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OPENVLA_CODE_ROOT="${PROJECT_ROOT}/openvla-oft_code"
CACHE_BASE_DIR="${CACHE_ROOT:-${MAIN_ROOT}/../.cache}"

LIBERO_BENCHMARK="${1:-libero}"
TASK_SUITE_NAME="${2:-libero_spatial}"
SELF_EVAL_MODE="${3:-${SELF_EVAL_MODE:-consistency}}"
if [[ "$SELF_EVAL_MODE" == "mad" ]]; then
    SELF_EVAL_MODE="mae-d"
fi
ATTENTION_EVAL_RATIOS="${ATTENTION_EVAL_RATIOS:-0.01}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-3}"
CONSISTENCY_REPEATS="${CONSISTENCY_REPEATS:-3}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.01}"
SAVE_ATTENTIONS="${SAVE_ATTENTIONS:-False}"

OFT_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${CACHE_BASE_DIR}/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial-object-goal-10/snapshots/638918f3d1c2e43a39a8a20772bdb8b91835e4b7}"
LIBERO_PRO_EVAL_CONFIG="${EVALUATION_CONFIG_PATH:-${MAIN_ROOT}/third_party/LIBERO_PRO/evaluation_config_swap.yaml}"

if [[ "$LIBERO_BENCHMARK" != "libero" && "$LIBERO_BENCHMARK" != "libero_pro" ]]; then
    echo "Unsupported LIBERO_BENCHMARK=$LIBERO_BENCHMARK"
    exit 1
fi

case "${TASK_SUITE_NAME}" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *)
        echo "Unsupported TASK_SUITE_NAME=$TASK_SUITE_NAME"
        exit 1
        ;;
esac

if [[ ! -d "$OFT_CHECKPOINT" ]]; then
    echo "Checkpoint not found: $OFT_CHECKPOINT"
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    DATE_STAMP="${DATE_STAMP:-$(date +%Y%m%d)}"
    SLURM_LOG_DIR="${PROJECT_ROOT}/logs/${DATE_STAMP}"
    LOCAL_LOG_DIR="${PROJECT_ROOT}/experiments/logs/${DATE_STAMP}"
    mkdir -p "$SLURM_LOG_DIR" "$LOCAL_LOG_DIR"
    LOG_PREFIX="${LIBERO_BENCHMARK}_${TASK_SUITE_NAME}_selfeval_${SELF_EVAL_MODE}"
    exec sbatch \
        --job-name="${LOG_PREFIX}" \
        --output="${SLURM_LOG_DIR}/${LOG_PREFIX}_%j.log" \
        --error="${SLURM_LOG_DIR}/${LOG_PREFIX}_%j.err" \
        --export=ALL,DATE_STAMP="${DATE_STAMP}",LOCAL_LOG_DIR="${LOCAL_LOG_DIR}" \
        "$0" "$@"
fi

export OPENVLA_OFT_ROOT="${PROJECT_ROOT}"
export VLA_SELFAWARE_CP_ROOT="${OPENVLA_OFT_ROOT}"  # backward-compatible alias
export OPENVLA_EVAL_RESULTS_ROOT="${OPENVLA_EVAL_RESULTS_ROOT:-${PROJECT_ROOT}/eval_results}"

CONDA_BASE_PATH="${CONDA_BASE:-$(conda info --base)}"
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-openvla}"

# shellcheck source=/dev/null
source "${MAIN_ROOT}/third_party/libero_env.sh"
setup_libero_env "${LIBERO_BENCHMARK}"

export OPENVLA_CODE_PATH="${OPENVLA_CODE_ROOT}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
cd "${PROJECT_ROOT}"

export DATE_STAMP="${DATE_STAMP:-$(date +%Y%m%d)}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${PROJECT_ROOT}/experiments/logs/${DATE_STAMP}}"
mkdir -p "$LOCAL_LOG_DIR"
# OFT code last so experiments.robot resolves to openvla-oft_code.
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${LIBERO_PATH}:${OPENVLA_CODE_PATH}"

export HF_HOME="${CACHE_BASE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

module load cuda/12.4 2>/dev/null || true
mkdir -p eval_results

COMMON_ARGS=(
  --pretrained_checkpoint "$OFT_CHECKPOINT"
  --task_suite_name "$TASK_SUITE_NAME"
  --libero_benchmark "$LIBERO_BENCHMARK"
  --enable_self_eval True
  --self_eval_mode "$SELF_EVAL_MODE"
  --save_attentions "${SAVE_ATTENTIONS}"
  --num_trials_per_task "$NUM_TRIALS_PER_TASK"
  --consistency_repeats_per_init "$CONSISTENCY_REPEATS"
  --action_noise_std "$ACTION_NOISE_STD"
  --attention_eval_ratios "$ATTENTION_EVAL_RATIOS"
  --local_log_dir "$LOCAL_LOG_DIR"
  --use_l1_regression True
  --num_images_in_input 2
  --use_proprio True
)

if [[ "$LIBERO_BENCHMARK" == "libero_pro" ]]; then
    python ./openvla-oft_code/experiments/robot/libero/run_multiprocess_libero_pro_eval.py \
      "${COMMON_ARGS[@]}" \
      --evaluation_config_path "$LIBERO_PRO_EVAL_CONFIG"
else
    python ./openvla-oft_code/experiments/robot/libero/run_multiprocess_libero_eval.py \
      "${COMMON_ARGS[@]}"
fi

RESULT_FILE="self_eval_consistency_scores.json"
if [[ "$SELF_EVAL_MODE" == "mae-d" ]]; then
    RESULT_FILE="online_MAE-D_scores.jsonl"
fi
echo "Self-eval output: ${OPENVLA_EVAL_RESULTS_ROOT}/${LIBERO_BENCHMARK}/<checkpoint>/${TASK_SUITE_NAME}/${DATE_STAMP}/${RUN_STAMP}/${RESULT_FILE}"
