#!/bin/bash -l

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --partition=a100

# Usage:
#   bash openvla_run_libero_self_eval.sh [libero|libero_reflect] [task_suite] [consistency|output_stats|mae-d]
# MAE-D online: SELF_EVAL_MODE=mae-d (writes online_MAE-D_scores.jsonl; no attention .pt save)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENVLA_ROOT="${SCRIPT_DIR}"
CACHE_BASE_DIR="${CACHE_ROOT:-${MAIN_ROOT}/../.cache}"
HF_HUB="${CACHE_BASE_DIR}/huggingface/hub"

LIBERO_BENCHMARK="${1:-libero}"
TASK_SUITE_NAME="${2:-libero_spatial}"
SELF_EVAL_MODE="${3:-${SELF_EVAL_MODE:-output_stats}}"
CONSISTENCY_REPEATS="${CONSISTENCY_REPEATS:-3}"
CONSISTENCY_METHOD="${CONSISTENCY_METHOD:-token_sample}"
SAVE_ATTENTIONS="${SAVE_ATTENTIONS:-False}"

if [[ "$LIBERO_BENCHMARK" != "libero" && "$LIBERO_BENCHMARK" != "libero_reflect" && "$LIBERO_BENCHMARK" != "libero_pro" ]]; then
    echo "Unsupported LIBERO_BENCHMARK=$LIBERO_BENCHMARK (expected: libero | libero_reflect)"
    exit 1
fi
if [[ "$LIBERO_BENCHMARK" == "libero_pro" ]]; then
    LIBERO_BENCHMARK="libero_reflect"
fi

if [[ "$SELF_EVAL_MODE" != "consistency" && "$SELF_EVAL_MODE" != "output_stats" && "$SELF_EVAL_MODE" != "mae-d" && "$SELF_EVAL_MODE" != "mad" ]]; then
    echo "Unsupported SELF_EVAL_MODE=$SELF_EVAL_MODE (expected: consistency | output_stats | mae-d)"
    exit 1
fi
if [[ "$SELF_EVAL_MODE" == "mad" ]]; then
    SELF_EVAL_MODE="mae-d"
fi

ATTENTION_EVAL_RATIOS="${ATTENTION_EVAL_RATIOS:-0.01}"

if [[ "$SELF_EVAL_MODE" == "consistency" ]]; then
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
    ACTION_SAMPLING_TEMPERATURE="${ACTION_SAMPLING_TEMPERATURE:-1.0}"
    if [[ "$CONSISTENCY_METHOD" == "outcome_repeat" && "$NUM_TRIALS_PER_TASK" == "50" ]]; then
        NUM_TRIALS_PER_TASK=3
    fi
else
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
    ACTION_SAMPLING_TEMPERATURE="${ACTION_SAMPLING_TEMPERATURE:-0.0}"
fi

case "${TASK_SUITE_NAME}" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *)
        echo "Unsupported TASK_SUITE_NAME=$TASK_SUITE_NAME"
        exit 1
        ;;
esac

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
if [[ -z "$PRETRAINED_CHECKPOINT" ]]; then
    PRETRAINED_CHECKPOINT="${HF_HUB}/models--openvla--openvla-7b-finetuned-libero-spatial/snapshots/fa5ae1e7509348889295bba8e08621d8b55e9baf"
fi

LIBERO_PRO_EVAL_CONFIG="${EVALUATION_CONFIG_PATH:-${MAIN_ROOT}/third_party/LIBERO-REFLECT/model_configs/evaluation_config_swap.yaml}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    DATE_STAMP="${DATE_STAMP:-$(date +%Y%m%d)}"
    SLURM_LOG_DIR="${OPENVLA_ROOT}/logs/${DATE_STAMP}"
    LOCAL_LOG_DIR="${OPENVLA_ROOT}/experiments/logs/${DATE_STAMP}"
    mkdir -p "$SLURM_LOG_DIR" "$LOCAL_LOG_DIR"
    LOG_PREFIX="${LIBERO_BENCHMARK}_${TASK_SUITE_NAME}_selfeval_${SELF_EVAL_MODE}"
    exec sbatch \
        --job-name="${LOG_PREFIX}" \
        --output="${SLURM_LOG_DIR}/${LOG_PREFIX}_%j.log" \
        --error="${SLURM_LOG_DIR}/${LOG_PREFIX}_%j.err" \
        --export=ALL,DATE_STAMP="${DATE_STAMP}",LOCAL_LOG_DIR="${LOCAL_LOG_DIR}" \
        "$0" "$@"
fi

export OPENVLA_EVAL_RESULTS_ROOT="${OPENVLA_EVAL_RESULTS_ROOT:-${OPENVLA_ROOT}/eval_results}"

CONDA_BASE_PATH="${CONDA_BASE:-$(conda info --base)}"
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-openvla-origin}"

# Shared LIBERO / LIBERO-PRO paths and isolated config directories.
# shellcheck source=/dev/null
source "${MAIN_ROOT}/third_party/libero_env.sh"
setup_libero_env "${LIBERO_BENCHMARK}"

export OPENVLA_CODE_PATH="${OPENVLA_ROOT}"
export TOKENIZERS_PARALLELISM=false
cd "${OPENVLA_ROOT}"

export DATE_STAMP="${DATE_STAMP:-$(date +%Y%m%d)}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${OPENVLA_ROOT}/experiments/logs/${DATE_STAMP}}"
mkdir -p "$LOCAL_LOG_DIR"
# Prepend OpenVLA so discrete-token code is not shadowed by LIBERO trees.
export PYTHONPATH="${OPENVLA_CODE_PATH}:${LIBERO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HOME="${CACHE_BASE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

module load cuda/12.4 2>/dev/null || true
mkdir -p eval_results

COMMON_ARGS=(
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
  --task_suite_name "$TASK_SUITE_NAME"
  --libero_benchmark "$LIBERO_BENCHMARK"
  --enable_self_eval True
  --self_eval_mode "$SELF_EVAL_MODE"
  --save_attentions "${SAVE_ATTENTIONS}"
  --num_trials_per_task "$NUM_TRIALS_PER_TASK"
  --consistency_repeats_per_init "$CONSISTENCY_REPEATS"
  --consistency_method "$CONSISTENCY_METHOD"
  --action_sampling_temperature "$ACTION_SAMPLING_TEMPERATURE"
  --attention_eval_ratios "$ATTENTION_EVAL_RATIOS"
  --local_log_dir "$LOCAL_LOG_DIR"
)

if [[ "$LIBERO_BENCHMARK" == "libero_reflect" ]]; then
    python ./experiments/robot/libero/run_libero_pro_eval.py \
      "${COMMON_ARGS[@]}" \
      --evaluation_config_path "$LIBERO_PRO_EVAL_CONFIG"
else
    python ./experiments/robot/libero/run_libero_eval.py \
      "${COMMON_ARGS[@]}"
fi

JSON_NAME="self_eval_output_stats_scores.json"
if [[ "$SELF_EVAL_MODE" == "consistency" ]]; then
    JSON_NAME="self_eval_consistency_scores.json"
elif [[ "$SELF_EVAL_MODE" == "mae-d" ]]; then
    JSON_NAME="online_MAE-D_scores.jsonl"
fi

echo "Self-eval JSON: ${OPENVLA_EVAL_RESULTS_ROOT}/${LIBERO_BENCHMARK}/<checkpoint>/${TASK_SUITE_NAME}/${DATE_STAMP}/${RUN_STAMP}/${JSON_NAME}"
