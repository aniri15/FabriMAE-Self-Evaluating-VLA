#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$OPENPI_ROOT/../.." && pwd)}"
if [[ -z "${LINK_ROOT:-}" ]]; then
  LINK_ROOT="$(find "$PROJECT_ROOT" -maxdepth 1 -type d -name '*_link' -print -quit)"
fi
if [[ -z "${LINK_ROOT:-}" ]]; then
  echo "Could not infer LINK_ROOT; set LINK_ROOT to the directory containing openpi_storage, libero, and hf-cache." >&2
  exit 2
fi
STORAGE_ROOT="${STORAGE_ROOT:-$LINK_ROOT/openpi_storage}"
LIBERO_ROOT="${LIBERO_ROOT:-$LINK_ROOT/libero/LIBERO-plus}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$LINK_ROOT/hf-cache}"
VENV="${VENV:-$OPENPI_ROOT/.venv}"

BENCHMARK="${1:-${BENCHMARK:-libero_plus}}"
ATTENTION_EVAL_METHOD_WAS_SET=0
if [[ -n "${ATTENTION_EVAL_METHOD:-}" ]]; then
  ATTENTION_EVAL_METHOD_WAS_SET=1
fi
if [[ $# -ge 2 ]]; then
  SUITE="$2"
fi
if [[ $# -ge 3 ]]; then
  ATTENTION_EVAL_METHOD="$3"
  ATTENTION_EVAL_METHOD_WAS_SET=1
fi
case "$BENCHMARK" in
  libero_plus|libero-plus|liberoplus) ;;
  *)
    echo "Unsupported BENCHMARK=$BENCHMARK; expected libero_plus" >&2
    exit 2
    ;;
esac

MODEL_NAME="${MODEL_NAME:-pi05_libero}"
SUITE="${SUITE:-libero_mix}"
TASK_START="${TASK_START:-0}"
NUM_TASKS="${NUM_TASKS:-5}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
if [[ -z "${MAX_STEPS:-}" ]]; then
  case "$SUITE" in
    libero_spatial) MAX_STEPS=660 ;;
    libero_object) MAX_STEPS=840 ;;
    libero_goal) MAX_STEPS=900 ;;
    libero_10) MAX_STEPS=1560 ;;
    *) MAX_STEPS=520 ;;
  esac
fi
SEED="${SEED:-7}"
PERTURBATION="${PERTURBATION:-all}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
PORT="${PORT:-$((18000 + RANDOM % 1000))}"
GPU_LOG_INTERVAL_SEC="${GPU_LOG_INTERVAL_SEC:-60}"
SAVE_ATTENTION_METRICS="${SAVE_ATTENTION_METRICS:-0}"
SAVE_ONLINE_MAE="${SAVE_ONLINE_MAE:-}"
ATTENTION_EVAL_METHOD="${ATTENTION_EVAL_METHOD:-none}"
TTS_MODE="${TTS_MODE:-none}"
TTS_NUM_CANDIDATES="${TTS_NUM_CANDIDATES:-4}"
TTS_BRANCH_RATIO="${TTS_BRANCH_RATIO:-0.4}"
TTS_BRANCH_NOISE_SCALE="${TTS_BRANCH_NOISE_SCALE:-0.1}"
TTS_SCORE_MODE="${TTS_SCORE_MODE:-mae}"
FLOW_MG_MASK="${FLOW_MG_MASK:-language}"
FLOW_MG_STEPS="${FLOW_MG_STEPS:-4,7,9}"
case "$ATTENTION_EVAL_METHOD" in
  none|"") ATTENTION_EVAL_INTERNAL_MODE="" ;;
  mae-c|mac) ATTENTION_EVAL_INTERNAL_MODE="mac" ;;
  mae-d|mad) ATTENTION_EVAL_INTERNAL_MODE="mad" ;;
  both) ATTENTION_EVAL_INTERNAL_MODE="both" ;;
  *)
    echo "Unknown ATTENTION_EVAL_METHOD=$ATTENTION_EVAL_METHOD; expected none, mae-c, mae-d, mac, mad, or both" >&2
    exit 2
    ;;
esac

if [[ "$ATTENTION_EVAL_INTERNAL_MODE" != "" ]]; then
  if [[ -z "$SAVE_ONLINE_MAE" && "$ATTENTION_EVAL_METHOD_WAS_SET" == "1" ]]; then
    SAVE_ONLINE_MAE=1
  fi
  if [[ -z "${ATTENTION_EVAL_MODE:-}" || "${ATTENTION_EVAL_MODE:-}" == "off" ]]; then
    ATTENTION_EVAL_MODE="$ATTENTION_EVAL_INTERNAL_MODE"
  fi
fi

if [[ "$TTS_MODE" == "independent" ]]; then
  if [[ "$TTS_SCORE_MODE" == "velocity_diff" ]]; then
    OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data/libero_plus_eval_tts_independent_flowmg_${FLOW_MG_MASK}}"
  elif [[ "$TTS_SCORE_MODE" == "mae_diff" ]]; then
    OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data/libero_plus_eval_tts_independent_flowmg_maediff_${FLOW_MG_MASK}}"
  elif [[ "$TTS_SCORE_MODE" == "mae_velocity_diff" ]]; then
    OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data/libero_plus_eval_tts_independent_flowmg_mae_vdiff_${FLOW_MG_MASK}}"
  else
    OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data/libero_plus_eval_tts_independent}"
  fi
  ATTENTION_EVAL_MODE="${ATTENTION_EVAL_MODE:-mac}"
  ATTENTION_EVAL_RATIOS="${ATTENTION_EVAL_RATIOS:-0.01}"
  TTS_SELECTION_RATIO="${TTS_SELECTION_RATIO:-0.01}"
elif [[ "$TTS_MODE" == "branch" ]]; then
  if [[ "$TTS_SCORE_MODE" == "velocity_diff" ]]; then
    OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data/libero_plus_eval_tts_branch_flowmg_${FLOW_MG_MASK}}"
  elif [[ "$TTS_SCORE_MODE" == "mae_diff" ]]; then
    OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data/libero_plus_eval_tts_branch_flowmg_maediff_${FLOW_MG_MASK}}"
  elif [[ "$TTS_SCORE_MODE" == "mae_velocity_diff" ]]; then
    OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data/libero_plus_eval_tts_branch_flowmg_mae_vdiff_${FLOW_MG_MASK}}"
  else
    OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data/libero_plus_eval_tts_branch}"
  fi
  ATTENTION_EVAL_MODE="${ATTENTION_EVAL_MODE:-mac}"
  ATTENTION_EVAL_RATIOS="${ATTENTION_EVAL_RATIOS:-0.01}"
  TTS_SELECTION_RATIO="${TTS_SELECTION_RATIO:-0.01}"
else
  OUT_ROOT="${OUT_ROOT:-$STORAGE_ROOT/data}"
  ATTENTION_EVAL_MODE="${ATTENTION_EVAL_MODE:-off}"
  ATTENTION_EVAL_RATIOS="${ATTENTION_EVAL_RATIOS:-0.01,0.05,0.1,0.5}"
  TTS_SELECTION_RATIO="${TTS_SELECTION_RATIO:-0.1}"
fi

case "$ATTENTION_EVAL_MODE" in
  off|mac|mad|both) ;;
  *)
    echo "Unknown ATTENTION_EVAL_MODE=$ATTENTION_EVAL_MODE; expected off, mac, mad, or both" >&2
    exit 2
    ;;
esac

SAVE_ONLINE_MAE="${SAVE_ONLINE_MAE:-0}"

if [[ "$SAVE_ONLINE_MAE" != "0" && "$SAVE_ONLINE_MAE" != "false" && "$SAVE_ONLINE_MAE" != "False" ]]; then
  if [[ "$ATTENTION_EVAL_METHOD" == "none" || "$ATTENTION_EVAL_METHOD" == "" ]]; then
    ATTENTION_EVAL_METHOD="mae-c"
  fi
  if [[ "$ATTENTION_EVAL_MODE" == "off" ]]; then
    case "$ATTENTION_EVAL_METHOD" in
      mae-c|mac) ATTENTION_EVAL_MODE="mac" ;;
      mae-d|mad) ATTENTION_EVAL_MODE="mad" ;;
      both) ATTENTION_EVAL_MODE="both" ;;
    esac
  fi
fi

case "$TTS_MODE" in
  none|independent|branch) ;;
  *)
    echo "Unknown TTS_MODE=$TTS_MODE; expected none, independent, or branch" >&2
    exit 2
    ;;
esac

case "$TTS_SCORE_MODE" in
  mae|velocity_diff|mae_diff|mae_velocity_diff) ;;
  *)
    echo "Unknown TTS_SCORE_MODE=$TTS_SCORE_MODE; expected mae, velocity_diff, mae_diff, or mae_velocity_diff" >&2
    exit 2
    ;;
esac

case "$FLOW_MG_MASK" in
  language|vision|language_vision) ;;
  *)
    echo "Unknown FLOW_MG_MASK=$FLOW_MG_MASK; expected language, vision, or language_vision" >&2
    exit 2
    ;;
esac

if [[ "$TTS_MODE" != "none" && "$ATTENTION_EVAL_MODE" == "off" ]]; then
  echo "TTS_MODE=$TTS_MODE requires ATTENTION_EVAL_MODE=mac, mad, or both" >&2
  exit 2
fi

if [[ "$ATTENTION_EVAL_MODE" != "off" && "$ATTENTION_EVAL_METHOD" == "none" ]]; then
  case "$ATTENTION_EVAL_MODE" in
    mac) ATTENTION_EVAL_METHOD="mae-c" ;;
    mad) ATTENTION_EVAL_METHOD="mae-d" ;;
    both) ATTENTION_EVAL_METHOD="both" ;;
  esac
fi
case "$ATTENTION_EVAL_METHOD" in
  mae-c|mac) ONLINE_MAE_OUTPUT_NAME="${ONLINE_MAE_OUTPUT_NAME:-online_MAE-C_scores.jsonl}" ;;
  mae-d|mad) ONLINE_MAE_OUTPUT_NAME="${ONLINE_MAE_OUTPUT_NAME:-online_MAE-D_scores.jsonl}" ;;
  *) ONLINE_MAE_OUTPUT_NAME="${ONLINE_MAE_OUTPUT_NAME:-online_MAE_scores.jsonl}" ;;
esac

case "$SAVE_ATTENTION_METRICS" in
  0|1|false|true|False|True) ;;
  *)
    echo "Unknown SAVE_ATTENTION_METRICS=$SAVE_ATTENTION_METRICS; expected 0 or 1" >&2
    exit 2
    ;;
esac

case "$SAVE_ONLINE_MAE" in
  0|1|false|true|False|True) ;;
  *)
    echo "Unknown SAVE_ONLINE_MAE=$SAVE_ONLINE_MAE; expected 0 or 1" >&2
    exit 2
    ;;
esac

case "$MODEL_NAME" in
  pi05_base)
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HF_CACHE_ROOT/pi05_base}"
    NORM_STATS_CHECKPOINT="${NORM_STATS_CHECKPOINT:-$HF_CACHE_ROOT/pi05_libero}"
    ;;
  pi05_libero)
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HF_CACHE_ROOT/pi05_libero}"
    NORM_STATS_CHECKPOINT="${NORM_STATS_CHECKPOINT:-$HF_CACHE_ROOT/pi05_libero}"
    ;;
  *)
    echo "Unknown MODEL_NAME=$MODEL_NAME; expected pi05_base or pi05_libero" >&2
    exit 2
    ;;
esac

source "$VENV/bin/activate"
cd "$OPENPI_ROOT"

export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$LIBERO_ROOT:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$OPENPI_ROOT/.uv-cache}"
export TMPDIR="${TMPDIR:-$OPENPI_ROOT/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false

IMAGEMAGICK_ROOT="${IMAGEMAGICK_ROOT:-}"
if [[ -n "$IMAGEMAGICK_ROOT" && -d "$IMAGEMAGICK_ROOT/lib" ]]; then
  export MAGICK_HOME="$IMAGEMAGICK_ROOT"
  export LD_LIBRARY_PATH="$IMAGEMAGICK_ROOT/lib:${LD_LIBRARY_PATH:-}"
  export PATH="$IMAGEMAGICK_ROOT/bin:$PATH"
fi

if [[ "$PERTURBATION" == "all" ]]; then
  OUT_DIR="$OUT_ROOT/$MODEL_NAME/$SUITE/tasks_${TASK_START}_$((TASK_START + NUM_TASKS - 1))_seed${SEED}"
else
  OUT_DIR="$OUT_ROOT/$MODEL_NAME/$SUITE/$PERTURBATION/tasks_${TASK_START}_$((TASK_START + NUM_TASKS - 1))_seed${SEED}"
fi
mkdir -p "$OUT_DIR"

echo "========== Runtime environment =========="
echo "Python:                 $(command -v python)"
echo "Python version:         $(python --version 2>&1)"
echo "Virtual environment:    ${VIRTUAL_ENV:-unset}"
echo "LIBERO root:            $LIBERO_ROOT"
echo "Checkpoint:             $CHECKPOINT_DIR"
echo "Norm stats checkpoint:  $NORM_STATS_CHECKPOINT"
echo "Output directory:       $OUT_DIR"
echo "Perturbation:           $PERTURBATION"
echo "Replan steps:           $REPLAN_STEPS"
echo "Environment wait steps: $NUM_STEPS_WAIT"
echo "Policy server port:     $PORT"
echo "GPU log interval:       ${GPU_LOG_INTERVAL_SEC}s"
echo "Attention eval mode:    $ATTENTION_EVAL_MODE"
echo "Attention eval method:  $ATTENTION_EVAL_METHOD"
echo "Attention eval ratios:  $ATTENTION_EVAL_RATIOS"
echo "Save attention metrics: $SAVE_ATTENTION_METRICS"
echo "Save online MAE:        $SAVE_ONLINE_MAE"
echo "Online MAE output:      $ONLINE_MAE_OUTPUT_NAME"
echo "TTS mode:               $TTS_MODE"
echo "TTS branch ratio:       $TTS_BRANCH_RATIO"
echo "TTS branch noise scale: $TTS_BRANCH_NOISE_SCALE"
echo "TTS candidates:         $TTS_NUM_CANDIDATES"
echo "TTS selection ratio:    $TTS_SELECTION_RATIO"
echo "TTS score mode:         $TTS_SCORE_MODE"
echo "Flow-MG mask:           $FLOW_MG_MASK"
echo "Flow-MG steps:          $FLOW_MG_STEPS"
python - <<'PY'
import torch

print(f"Torch version:          {torch.__version__}")
print(f"Torch CUDA version:     {torch.version.cuda}")
print(f"CUDA available:         {torch.cuda.is_available()}")
print(f"CUDA device count:      {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        memory_gib = props.total_memory / 1024**3
        print(f"CUDA device {index}:          {props.name} ({memory_gib:.1f} GiB)")
PY
echo "========================================="
echo

GPU_LOG="$OUT_DIR/gpu_usage.csv"
echo "timestamp,index,name,temperature_gpu_c,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w" >"$GPU_LOG"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits >>"$GPU_LOG" 2>&1 || true
    sleep "$GPU_LOG_INTERVAL_SEC"
  done
) &
GPU_MONITOR_PID=$!
echo "GPU monitoring:         PID $GPU_MONITOR_PID -> $GPU_LOG"
echo

SERVER_START_EPOCH="$(date +%s)"
echo "[$(date --iso-8601=seconds)] Starting policy server..."
SERVER_ARGS=(
  python scripts/serve_policy_with_norm_stats.py
  --config pi05_libero
  --checkpoint-dir "$CHECKPOINT_DIR"
  --norm-stats-checkpoint "$NORM_STATS_CHECKPOINT"
  --port "$PORT"
)
if [[ "$ATTENTION_EVAL_MODE" != "off" ]]; then
  SERVER_ARGS+=(
    --attention-eval
    --attention-eval-mode "$ATTENTION_EVAL_MODE"
    --attention-eval-method "$ATTENTION_EVAL_METHOD"
    --attention-eval-ratios "$ATTENTION_EVAL_RATIOS"
    --tts-mode "$TTS_MODE"
    --tts-num-candidates "$TTS_NUM_CANDIDATES"
    --tts-selection-ratio "$TTS_SELECTION_RATIO"
    --tts-branch-ratio "$TTS_BRANCH_RATIO"
    --tts-branch-noise-scale "$TTS_BRANCH_NOISE_SCALE"
    --tts-score-mode "$TTS_SCORE_MODE"
    --flow-mg-mask "$FLOW_MG_MASK"
    --flow-mg-steps "$FLOW_MG_STEPS"
  )
fi
"${SERVER_ARGS[@]}" >"$OUT_DIR/policy_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] Stopping policy server PID $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python - <<PY
import time
import urllib.request
port = int("$PORT")
url = f"http://127.0.0.1:{port}/healthz"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(5)
raise SystemExit("policy server did not become healthy")
PY

SERVER_READY_EPOCH="$(date +%s)"
echo "[$(date --iso-8601=seconds)] Policy server is healthy (PID $SERVER_PID, startup $((SERVER_READY_EPOCH - SERVER_START_EPOCH))s)"
echo "Policy server log: $OUT_DIR/policy_server.log"
echo
echo "========== Evaluation started =========="

EVAL_START_EPOCH="$(date +%s)"
EVAL_ARGS=(
  python scripts/eval_libero_plus_policy.py
  --libero-root "$LIBERO_ROOT" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --suite "$SUITE" \
  --perturbation "$PERTURBATION" \
  --task-start "$TASK_START" \
  --num-tasks "$NUM_TASKS" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --max-steps "$MAX_STEPS" \
  --replan-steps "$REPLAN_STEPS" \
  --num-steps-wait "$NUM_STEPS_WAIT" \
  --seed "$SEED" \
  --attention-eval-mode "$ATTENTION_EVAL_MODE" \
  --attention-eval-method "$ATTENTION_EVAL_METHOD" \
  --attention-eval-ratios "$ATTENTION_EVAL_RATIOS" \
  --online-mae-output-name "$ONLINE_MAE_OUTPUT_NAME" \
  --tts-mode "$TTS_MODE" \
  --tts-num-candidates "$TTS_NUM_CANDIDATES" \
  --tts-selection-ratio "$TTS_SELECTION_RATIO" \
  --tts-branch-ratio "$TTS_BRANCH_RATIO" \
  --tts-branch-noise-scale "$TTS_BRANCH_NOISE_SCALE" \
  --tts-score-mode "$TTS_SCORE_MODE" \
  --flow-mg-mask "$FLOW_MG_MASK" \
  --flow-mg-steps "$FLOW_MG_STEPS" \
  --out-dir "$OUT_DIR"
)
case "$SAVE_ATTENTION_METRICS" in
  1|true|True)
    EVAL_ARGS+=(--save-attention-metrics)
    ;;
esac
case "$SAVE_ONLINE_MAE" in
  1|true|True)
    EVAL_ARGS+=(--save-online-mae)
    ;;
esac
"${EVAL_ARGS[@]}"
EVAL_END_EPOCH="$(date +%s)"

echo "========== Evaluation completed =========="
echo "Evaluation elapsed: $((EVAL_END_EPOCH - EVAL_START_EPOCH))s"
echo "Summary:            $OUT_DIR/summary.json"
echo "Episode results:    $OUT_DIR/episodes.csv"
echo "Policy server log:  $OUT_DIR/policy_server.log"
echo "GPU usage log:      $GPU_LOG"
case "$SAVE_ATTENTION_METRICS" in
  1|true|True)
  echo "Attention metrics:  $OUT_DIR/attention_metrics.jsonl"
    ;;
esac
case "$SAVE_ONLINE_MAE" in
  1|true|True)
  echo "Online MAE scores: $OUT_DIR/$ONLINE_MAE_OUTPUT_NAME"
    ;;
esac
