#!/usr/bin/env python3
"""
StarVLA LIBERO PRO 评测脚本 - 按照官方两步流程启动：
1) starVLA 环境跑 policy server（后台）
2) LIBERO PRO 环境跑 eval client（前台，不拆分 task）

支持的模型：
- StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 (默认，4合1模型)
- OpenVLA-OFT task-specific models (可选，用于对比)

使用方法：
  # 直接运行（需要在 GPU 节点）
  python submit_libero_pro_eval.py
  
  # 通过 SLURM 提交作业
  python submit_libero_pro_eval.py --slurm
  
  # 指定任务套件
  python submit_libero_pro_eval.py --suite libero_spatial --slurm
  
  # 使用 LIBERO PRO 扰动（自动查找配置文件，推荐）
  python submit_libero_pro_eval.py --suite libero_goal --perturbation env --slurm
  python submit_libero_pro_eval.py --suite libero_goal --perturbation swap --slurm
  python submit_libero_pro_eval.py --suite libero_goal --perturbation object --slurm
  python submit_libero_pro_eval.py --suite libero_goal --perturbation lang --slurm
  python submit_libero_pro_eval.py --suite libero_goal --perturbation task --slurm
  
  # 手动指定配置文件路径（如果自动查找失败）
  python submit_libero_pro_eval.py --suite libero_goal --evaluation-config-path /path/to/config.yaml --slurm
  
  # 完整参数
  python submit_libero_pro_eval.py --suite libero_goal --model oft --num-trials 50 --perturbation env --slurm
"""

import os
import subprocess
import sys
import time
import argparse
from pathlib import Path

# ------------------------------------------------------------------------------
# 基本路径与环境
# ------------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_MAIN_ROOT = _SCRIPT_DIR.parent
PROJECT_ROOT = _MAIN_ROOT
CONDA_BASE = Path(os.environ.get("CONDA_BASE", "")) if os.environ.get("CONDA_BASE") else None
STARVLA_ENV = os.environ.get("STARVLA_ENV", "starVLA")
LIBERO_ENV = os.environ.get("LIBERO_ENV", "libero_pro")

STARVLA_CODE_ROOT = _SCRIPT_DIR
LIBERO_ROOT = _MAIN_ROOT / "third_party" / "LIBERO"
LIBERO_PRO_DIR = Path(os.environ.get("LIBERO_PRO_DIR", str(_MAIN_ROOT / "third_party" / "LIBERO_PRO")))

# Perturbation Mappings (Short name -> Config Key)
PERTURBATION_MAP = {
    "env": "use_environment",
    "swap": "use_swap",
    "object": "use_object",
    "lang": "use_language",
    "task": "use_task"
}

# Number of tasks per suite 
TASKS_PER_SUITE = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

# Port assignments per suite (to avoid conflicts when running on same node)
PORT_PER_SUITE = {
    "libero_spatial": 5694,
    "libero_object": 5695,
    "libero_goal": 5696,
    "libero_10": 5697,
    "libero_90": 5698,
}

# StarVLA Checkpoint (4-in-1 model, can be used for all suites)
def _checkpoint_from_env(env_name: str) -> str:
    return os.environ.get(env_name, os.environ.get("PRETRAINED_CHECKPOINT", ""))


STARVLA_4IN1_CHECKPOINT = {
    "oft": _checkpoint_from_env("CHECKPOINT_OFT"),
    "fast": _checkpoint_from_env("CHECKPOINT_FAST"),
    "groot": _checkpoint_from_env("CHECKPOINT_GROOT"),
    "pi": _checkpoint_from_env("CHECKPOINT_PI"),
}


# Model type names for display
MODEL_NAMES = {
    "oft": "Qwen2.5-VL-OFT-LIBERO-4in1",
    "fast": "Qwen2.5-VL-FAST-LIBERO-4in1",
    "groot": "Qwen2.5-VL-GR00T-LIBERO-4in1",
    "pi": "Qwen3-VL-PI-LIBERO-4in1",
}

MODEL_DIR_NAMES = {
    "oft": "qwenoft",
    "fast": "qwenfast",
    "groot": "qwengroot",
    "pi": "qwenpi",
}

# 上游的两个脚本（本脚本会生成带本地路径的副本来运行）
SERVER_SH = STARVLA_CODE_ROOT / "examples/LIBERO/eval_files/run_policy_server.sh"
CLIENT_PY = STARVLA_CODE_ROOT / "examples/LIBERO/eval_files/eval_libero.py"

# SLURM 作业模板
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=starvla-{suite}
#SBATCH --output={log_path_out}
#SBATCH --error={log_path_err}
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100

echo "======================================================"
echo "StarVLA LIBERO Evaluation via SLURM"
echo "Job launched at: $(date)"
echo "Suite: {suite}"
echo "Model: {model_type}"
echo "======================================================"

# Run the evaluation script
cd {project_root}
export MODEL_TYPE="{model_type_key}"
export CKPT_PATH="{ckpt_path}"
export TASK_SUITE="{suite}"
export NUM_TRIALS="{num_trials}"
export GPU_ID="0"
export PORT="{port}"
export WAIT_SECS="{wait_secs}"
export RUN_STAMP="{run_stamp}"        # 使用固定的时间戳
export DATE_STAMP="{date_stamp}"
export EVALUATION_CONFIG_PATH="{evaluation_config_path}"  # LIBERO PRO evaluation config path
export PERTURBATION_TYPE="{perturbation_dir}"  # Pass perturbation type to child script
export ATTENTION_EVAL_MODE="{ATTENTION_EVAL_MODE}"
export ATTENTION_EVAL_METHOD="{ATTENTION_EVAL_METHOD}"
export ATTENTION_EVAL_RATIOS="{ATTENTION_EVAL_RATIOS}"
export ACTION_NOISE_STD="{ACTION_NOISE_STD}"

python {script_path}
nvidia-smi
echo "Job finished at: $(date)"
"""

# ------------------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="StarVLA LIBERO PRO Evaluation")
    parser.add_argument("--slurm", action="store_true", help="通过 SLURM 提交作业而不是直接运行")
    parser.add_argument("--suite", type=str, default=None, choices=list(TASKS_PER_SUITE.keys()),
                        help="任务套件 (默认: libero_goal)")
    parser.add_argument("--model", type=str, default=None, help="model type (oft, fast, groot, pi)")
    parser.add_argument("--num-trials", type=int, default=None, help="每个任务的试验次数 (默认: 50)")
    parser.add_argument("--gpu-id", type=str, default=None, help="GPU ID (默认: 0)")
    parser.add_argument("--port", type=int, default=None, help="服务器端口 (默认: 5694)")
    parser.add_argument("--wait-secs", type=int, default=None, help="等待服务器启动的秒数 (默认: 90)")
    parser.add_argument("--evaluation-config-path", type=str, default=None,
                        help="LIBERO PRO evaluation config YAML file path (for perturbations)")
    parser.add_argument("--pert", type=str, default=None,
                        choices=["env", "swap", "object", "lang", "task"],
                        help="扰动类型 (env/swap/object/lang/task)，自动查找配置文件")
    parser.add_argument("--attention-eval-mode", type=str, default=None, choices=["save", "eval"],
                        help="在线 attention 评估模式：save(默认关闭在线评估)/eval(在线计算，不保存attention)")
    parser.add_argument("--attention-eval-method", type=str, default=None, choices=["mae-c", "mae-d", "mac", "mad"],
                        help="在线评估方法，Pi建议使用 mae-c（兼容 mac/mad 别名）")
    parser.add_argument("--attention-eval-ratios", type=str, default=None,
                        help="在线评估比例，如 0.01")
    parser.add_argument("--action-noise-std", type=float, default=None,
                        help="在动作前6维注入高斯噪声的标准差，默认0.0")
    parser.add_argument("--consistency-repeats-per-init", type=int, default=None,
                        help="每个固定初始状态重复评测次数，默认1")
    return parser.parse_args()

args = parse_args()

# 立即输出，确保在SLURM环境中能看到
import sys
sys.stdout.flush()
sys.stderr.flush()

print(f"[DEBUG] Script started, args.slurm={args.slurm}", flush=True)
print(f"[DEBUG] args.pert={args.pert}", flush=True)
print(f"[DEBUG] Environment variables:", flush=True)
print(f"  MODEL_TYPE={os.environ.get('MODEL_TYPE', 'NOT SET')}", flush=True)
print(f"  TASK_SUITE={os.environ.get('TASK_SUITE', 'NOT SET')}", flush=True)
print(f"  EVALUATION_CONFIG_PATH={os.environ.get('EVALUATION_CONFIG_PATH', 'NOT SET')}", flush=True)
print(f"  PERTURBATION_TYPE={os.environ.get('PERTURBATION_TYPE', 'NOT SET')}", flush=True)

# ------------------------------------------------------------------------------
# 可调参数（命令行参数优先，然后环境变量，最后默认值）
# ------------------------------------------------------------------------------
# 获取模型类型和检查点路径
MODEL_TYPE = args.model or os.environ.get("MODEL_TYPE", "oft")  # 默认使用 oft
if MODEL_TYPE not in ["oft", "fast", "groot", "pi"]:
    print(f"❌ MODEL_TYPE 必须是 oft, fast, groot, pi 之一，当前值: {MODEL_TYPE}")
    print(f"   使用 --model <type> 指定模型类型")
    sys.exit(1)

# 从字典获取对应的检查点路径
CKPT_PATH = STARVLA_4IN1_CHECKPOINT[MODEL_TYPE]

# 其他参数
TASK_SUITE = args.suite or os.environ.get("TASK_SUITE", "libero_goal")
NUM_TRIALS = args.num_trials or int(os.environ.get("NUM_TRIALS", "50"))
HOST = os.environ.get("HOST", "127.0.0.1")
GPU_ID = args.gpu_id or os.environ.get("GPU_ID", "0")
WAIT_SECS = args.wait_secs or int(os.environ.get("WAIT_SECS", "90"))
ATTENTION_EVAL_MODE = args.attention_eval_mode or os.environ.get("ATTENTION_EVAL_MODE", "save")
ATTENTION_EVAL_METHOD = args.attention_eval_method or os.environ.get("ATTENTION_EVAL_METHOD", "mae-c")
ATTENTION_EVAL_RATIOS = args.attention_eval_ratios or os.environ.get("ATTENTION_EVAL_RATIOS", "0.01")
ACTION_NOISE_STD = args.action_noise_std if args.action_noise_std is not None else float(os.environ.get("ACTION_NOISE_STD", "0.0"))
CONSISTENCY_REPEATS_PER_INIT = (
    args.consistency_repeats_per_init
    if args.consistency_repeats_per_init is not None
    else int(os.environ.get("CONSISTENCY_REPEATS_PER_INIT", "1"))
)

# 处理评估配置路径：优先使用 --perturbation，然后是 --evaluation-config-path
EVALUATION_CONFIG_PATH = None
if args.pert:
    # 根据扰动类型自动构建配置文件路径
    config_filename = f"eval_config_{TASK_SUITE}_{args.pert}.yaml"
    config_path = LIBERO_PRO_DIR / "generated_configs" / config_filename
    
    if config_path.exists():
        EVALUATION_CONFIG_PATH = str(config_path)
        print(f"✓ 自动找到配置文件: {EVALUATION_CONFIG_PATH}")
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        print(f"   请检查 LIBERO_PRO/generated_configs/ 目录下是否有 {config_filename}")
        sys.exit(1)
elif args.evaluation_config_path:
    EVALUATION_CONFIG_PATH = args.evaluation_config_path
else:
    EVALUATION_CONFIG_PATH = os.environ.get("EVALUATION_CONFIG_PATH", None)

if TASK_SUITE not in TASKS_PER_SUITE:
    print(f"❌ TASK_SUITE 必须是 {list(TASKS_PER_SUITE.keys())} 之一")
    sys.exit(1)

# 自动为每个任务套件分配不同的端口（避免同节点上的端口冲突）
DEFAULT_PORT = PORT_PER_SUITE.get(TASK_SUITE, 5694)
PORT = args.port or int(os.environ.get("PORT", str(DEFAULT_PORT)))

# ------------------------------------------------------------------------------
# 日志与输出目录
# ------------------------------------------------------------------------------
# 如果是 SLURM 作业重新运行，使用传入的时间戳；否则创建新的
RUN_STAMP = os.environ.get("RUN_STAMP") or time.strftime("%Y%m%d_%H%M%S")
date_stamp = os.environ.get("DATE_STAMP") or time.strftime("%Y%m%d")
job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID") or RUN_STAMP
model_dir = MODEL_DIR_NAMES.get(MODEL_TYPE, f"qwen{MODEL_TYPE}")

# 统一的时间戳目录（所有日志和输出都使用这个）
# 如果提供了扰动类型，在路径中包含它；否则从 EVALUATION_CONFIG_PATH 推断，或使用 "standard"
perturbation_dir = args.pert
print(f"[DEBUG] perturbation_dir after args.pert: {perturbation_dir}", flush=True)

# 如果 args.pert 为空，尝试从环境变量获取
if not perturbation_dir:
    perturbation_dir = os.environ.get("PERTURBATION_TYPE", None)
    print(f"[DEBUG] perturbation_dir from env: {perturbation_dir}", flush=True)

# 如果还是没有，从配置文件路径推断
if not perturbation_dir and EVALUATION_CONFIG_PATH:
    # 从配置文件路径推断扰动类型：eval_config_{suite}_{pert}.yaml
    import re
    config_basename = Path(EVALUATION_CONFIG_PATH).stem
    print(f"[DEBUG] 尝试从配置文件路径推断: {config_basename}", flush=True)
    match = re.search(r'evaluation_config_(env|swap|obj|lan|task)', config_basename)
    if match:
        perturbation_dir = match.group(1)
        print(f"[DEBUG] 从配置文件路径推断扰动类型: {perturbation_dir}", flush=True)
    else:
        print(f"[DEBUG] 无法从配置文件路径推断扰动类型", flush=True)

# 如果还是没有，使用默认值
if not perturbation_dir:
    perturbation_dir = "standard"
    print(f"[DEBUG] 使用默认扰动类型: {perturbation_dir}", flush=True)

print(f"[DEBUG] 最终 perturbation_dir: {perturbation_dir}", flush=True)
TIMESTAMP_DIR = STARVLA_CODE_ROOT / "logs" / model_dir / "libero" / date_stamp / str(job_id)
TIMESTAMP_DIR.mkdir(parents=True, exist_ok=True)

LOG_ROOT = TIMESTAMP_DIR  # Server 日志和 SLURM 日志都在这里

# 解析 checkpoint 路径以确定输出目录和文件夹名
ckpt_path = Path(CKPT_PATH)
if "/checkpoints/" in CKPT_PATH:
    # starVLA 格式: .../run_dir/checkpoints/xxx.pt
    folder_name = ckpt_path.stem  # 去掉 .pt 后缀
else:
    # HuggingFace 格式: .../snapshots/hash/xxx.pt
    folder_name = f"{ckpt_path.parents[1].name}_{ckpt_path.stem}"  # hash_filename

# Video output path format:
# results/libero/{perturbation_type}/{model_type}/{task_suite_name}/{date}/{date_timestamp}
video_out_path = str(STARVLA_CODE_ROOT / "results" / "libero" / perturbation_dir / MODEL_TYPE  / TASK_SUITE / date_stamp / RUN_STAMP)
client_log_path = str(TIMESTAMP_DIR / "logs")
Path(video_out_path).mkdir(parents=True, exist_ok=True)
Path(client_log_path).mkdir(parents=True, exist_ok=True)

# 获取模型显示名称
model_type = MODEL_NAMES.get(MODEL_TYPE, f"Unknown-{MODEL_TYPE}")

print("=" * 80, flush=True)
print(f"=== StarVLA LIBERO PRO Evaluation ===", flush=True)
print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"Model: {model_type} ({MODEL_TYPE})", flush=True)
print(f"Suite: {TASK_SUITE} ({TASKS_PER_SUITE[TASK_SUITE]} tasks)", flush=True)
print(f"Checkpoint: {CKPT_PATH}", flush=True)
print(f"Server port: {PORT}", flush=True)
print(f"GPU: {GPU_ID}", flush=True)
print(f"Trials per task: {NUM_TRIALS}", flush=True)
print(f"Action noise std: {ACTION_NOISE_STD}", flush=True)
print(f"Consistency repeats per init: {CONSISTENCY_REPEATS_PER_INIT}", flush=True)
print(f"Attention eval mode: {ATTENTION_EVAL_MODE}", flush=True)
if ATTENTION_EVAL_MODE == "eval":
    print(f"Attention eval method: {ATTENTION_EVAL_METHOD}", flush=True)
    print(f"Attention eval ratios: {ATTENTION_EVAL_RATIOS}", flush=True)
if args.pert:
    print(f"Perturbation type: {args.pert}", flush=True)
    print(f"LIBERO PRO config: {EVALUATION_CONFIG_PATH}", flush=True)
elif EVALUATION_CONFIG_PATH:
    print(f"LIBERO PRO config: {EVALUATION_CONFIG_PATH}", flush=True)
else:
    print(f"LIBERO PRO config: None (standard LIBERO evaluation)", flush=True)
print(f"\nOutput locations:", flush=True)
print(f"  Server logs: {LOG_ROOT}", flush=True)
print(f"  Client logs: {client_log_path}", flush=True)
print(f"  Videos: {video_out_path}", flush=True)
print("=" * 80, flush=True)

# ------------------------------------------------------------------------------
# SLURM 提交模式
# ------------------------------------------------------------------------------
if args.slurm:
    print("\n[SLURM Mode] 生成并提交 SLURM 作业...")
    
    # 创建 SLURM 作业目录
    slurm_jobs_dir = PROJECT_ROOT / "generated_slurm_jobs"
    slurm_jobs_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志路径（提交阶段只知道日期，作业内会落到 .../<job_id>/）
    slurm_logs_dir = STARVLA_CODE_ROOT / "logs" / model_dir / "libero" / date_stamp
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path_out = slurm_logs_dir / f"slurm_{TASK_SUITE}_%j.out"
    log_path_err = slurm_logs_dir / f"slurm_{TASK_SUITE}_%j.err"
    
    # 生成 SLURM 脚本
    job_filename = f"starvla_eval_{TASK_SUITE}_{RUN_STAMP}.sh"
    job_path = slurm_jobs_dir / job_filename
    
    script_content = SLURM_TEMPLATE.format(
        suite=TASK_SUITE,
        log_path_out=log_path_out,
        log_path_err=log_path_err,
        model_type=model_type,
        model_type_key=MODEL_TYPE,
        project_root=PROJECT_ROOT,
        ckpt_path=CKPT_PATH,
        num_trials=NUM_TRIALS,
        port=PORT,
        wait_secs=WAIT_SECS,
        run_stamp=RUN_STAMP,
        date_stamp=date_stamp,
        evaluation_config_path=EVALUATION_CONFIG_PATH or "",
        perturbation_dir=perturbation_dir,
        ATTENTION_EVAL_MODE=ATTENTION_EVAL_MODE,
        ATTENTION_EVAL_METHOD=ATTENTION_EVAL_METHOD,
        ATTENTION_EVAL_RATIOS=ATTENTION_EVAL_RATIOS,
        ACTION_NOISE_STD=ACTION_NOISE_STD,
        script_path=Path(__file__).absolute()
    )
    
    with open(job_path, 'w') as f:
        f.write(script_content)
    
    print(f"✓ SLURM 脚本已生成: {job_path}")
    
    # 提交作业
    try:
        result = subprocess.run(["sbatch", str(job_path)], capture_output=True, text=True, check=True)
        job_id_line = result.stdout.strip()
        # Extract job ID from "Submitted batch job 12345"
        job_id = job_id_line.split()[-1] if "batch job" in job_id_line else job_id_line
        
        print(f"✓ 作业已提交: {job_id_line}")
        final_job_dir = STARVLA_CODE_ROOT / "logs" / model_dir / "libero" / date_stamp / str(job_id)
        print(f"\n作业内日志目录:")
        print(f"  {final_job_dir}/")
        print(f"\n监控作业:")
        print(f"  squeue -u $USER")
        print(f"  tail -f {slurm_logs_dir}/slurm_{TASK_SUITE}_{job_id}.out")
        print(f"\n查看所有日志:")
        print(f"  ls -lh {final_job_dir}/")
        print("\n" + "=" * 80)
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"❌ 提交失败: {e.stderr}")
        sys.exit(1)

# ------------------------------------------------------------------------------
# 检查 GPU 可用性
# ------------------------------------------------------------------------------
print("\n[GPU Check] 验证 GPU 分配...", flush=True)
try:
    import torch
    print(f"[GPU Check] PyTorch version: {torch.__version__}", flush=True)
    cuda_available = torch.cuda.is_available()
    print(f"[GPU Check] CUDA available: {cuda_available}", flush=True)
    device_count = torch.cuda.device_count()
    print(f"[GPU Check] Device count: {device_count}", flush=True)
    
    if cuda_available and device_count > 0:
        gpu_id_int = int(GPU_ID)
        if gpu_id_int >= device_count:
            print(f"⚠️  警告: 请求 GPU {GPU_ID}，但只有 {device_count} 个 GPU 可用 (0-{device_count-1})", flush=True)
            print(f"   自动使用 GPU 0", flush=True)
            GPU_ID = "0"
        
        # 显示 GPU 信息
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"✓ GPU {i}: {props.name} ({mem_gb:.1f} GB)", flush=True)
        
        print(f"✓ 将使用 GPU {GPU_ID} 运行模型", flush=True)
        
        # 测试 GPU 是否真的可访问
        test_tensor = torch.zeros(1).cuda(int(GPU_ID))
        print(f"✓ GPU {GPU_ID} 可访问测试通过", flush=True)
        
    else:
        print("❌ 错误: 未检测到 GPU!", flush=True)
        print("   可能原因:", flush=True)
        print("   1. 未在 GPU 节点上运行", flush=True)
        print("   2. 未通过 SLURM 分配 GPU (需要 --gres=gpu:N)", flush=True)
        print("   3. CUDA/PyTorch 配置问题", flush=True)
        print("\n   请使用以下方式之一:", flush=True)
        print("   - 交互式: srun --gres=gpu:1 --pty bash", flush=True)
        print("   - SLURM 脚本: #SBATCH --gres=gpu:1", flush=True)
        sys.exit(1)
        
except Exception as e:
    import traceback
    print(f"❌ GPU 检查失败: {e}", flush=True)
    print(f"   详细错误: {traceback.format_exc()}", flush=True)
    print("   继续运行，但可能会失败...", flush=True)

print(flush=True)

# ------------------------------------------------------------------------------
# 生成本地化脚本（保持"跑这两个 sh"但替换路径/环境/ckpt）
# ------------------------------------------------------------------------------
print("[Script Generation] 生成本地化脚本...", flush=True)
server_local = LOG_ROOT / "run_policy_server_local.sh"
client_local = LOG_ROOT / "eval_libero_pro_local.sh"

server_local.write_text(
    f"""#!/bin/bash
set -eo pipefail

export http_proxy="${HTTP_PROXY:-}"
export https_proxy="${HTTPS_PROXY:-}"

# 1. Initialize Conda
CONDA_BASE_PATH="${CONDA_BASE}"
source "${{CONDA_BASE_PATH}}/etc/profile.d/conda.sh"
conda activate {STARVLA_ENV}

# 2. Load CUDA module
if command -v module >/dev/null 2>&1; then
  module load cuda/12.2 >/dev/null 2>&1 || echo "[WARN] cuda/12.2 module not found, skipping"
fi

# 3. Set LD_LIBRARY_PATH for flash_attn compatibility
# Ensure CUDA libraries are in the path for flash_attn
# export LD_LIBRARY_PATH="${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH}}"
# export LD_LIBRARY_PATH="${REMOTE_PATH_REMOVED}"

# 4. Set environment variables
cd "{STARVLA_CODE_ROOT}"
export PYTHONPATH="$(pwd):${{PYTHONPATH}}"

# Use local HuggingFace cache (compute nodes have no internet)
export HF_HOME="${CACHE_ROOT}/"
export HF_HUB_CACHE="${{HF_HOME}}/hub"
export TRANSFORMERS_CACHE="${{HF_HOME}}"
# export HF_HUB_OFFLINE=1  # Force offline mode to prevent network requests

# 5. Run server
your_ckpt="{CKPT_PATH}"
gpu_id="{GPU_ID}"
port="{PORT}"
CUDA_VISIBLE_DEVICES=${{gpu_id}} python deployment/model_server/server_policy.py \\
    --ckpt_path ${{your_ckpt}} \\
    --port ${{port}} \\
    --use_bf16
"""
)
server_local.chmod(0o755)
print(f"[Script Generation] Server script: {server_local}", flush=True)

client_local.write_text(
    f"""#!/bin/bash
set -eo pipefail

# 1. Initialize Conda
CONDA_BASE_PATH="${CONDA_BASE}"
source "${{CONDA_BASE_PATH}}/etc/profile.d/conda.sh"
conda activate {LIBERO_ENV}

# 2. Load CUDA module (optional)
if command -v module >/dev/null 2>&1; then
  module load cuda/12.2 >/dev/null 2>&1 || echo "[WARN] cuda/12.2 module not found, skipping"
fi

# 3. Set environment variables
cd "{STARVLA_CODE_ROOT}"
export LIBERO_PATH="{LIBERO_ROOT}"
export LIBERO_HOME="{LIBERO_ROOT}"
export LIBERO_CONFIG_PATH=${{LIBERO_HOME}}/libero
export STARVLA_CODE_PATH="{STARVLA_CODE_ROOT}"
export PYTHONPATH="${{PYTHONPATH}}:${{LIBERO_PATH}}:${{STARVLA_CODE_PATH}}"
export PYTHONPATH="$(pwd):${{PYTHONPATH}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Set date and timestamp for video path (use environment variables if set, otherwise generate)
export DATE_STAMP="${{DATE_STAMP:-$(date +%Y%m%d)}}"
export RUN_STAMP="${{RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}}"

# Ensure LIBERO config exists to avoid interactive prompts
mkdir -p "${{LIBERO_CONFIG_PATH}}"
if [ ! -f "${{LIBERO_CONFIG_PATH}}/config.yaml" ]; then
  cat > "${{LIBERO_CONFIG_PATH}}/config.yaml" <<EOF
benchmark_root: ${{LIBERO_HOME}}/libero/libero
bddl_files: ${{LIBERO_HOME}}/libero/libero/bddl_files
init_states: ${{LIBERO_HOME}}/libero/libero/init_files
datasets: ${{LIBERO_HOME}}/datasets
assets: ${{LIBERO_HOME}}/libero/libero/assets
EOF
  echo "[INFO] Created LIBERO config file: ${{LIBERO_CONFIG_PATH}}/config.yaml"
fi

# 4. Run evaluation
host="{HOST}"
base_port="{PORT}"
unnorm_key="franka"
your_ckpt="{CKPT_PATH}"

LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p ${{LOG_DIR}}

task_suite_name="{TASK_SUITE}"
num_trials_per_task="{NUM_TRIALS}"
perturbation_type="{perturbation_dir}"
# Video path format: results/libero_pro/{{perturbation_type}}/{{model_type}}/{{task_suite_name}}/{{date}}/{{date_timestamp}}
# Use environment variables DATE_STAMP and RUN_STAMP that are set in SLURM template
video_out_path="results/libero_pro/{MODEL_TYPE}"
evaluation_config_path="{EVALUATION_CONFIG_PATH}"
attention_eval_mode="{ATTENTION_EVAL_MODE}"
attention_eval_method="{ATTENTION_EVAL_METHOD}"
attention_eval_ratios="{ATTENTION_EVAL_RATIOS}"
action_noise_std="{ACTION_NOISE_STD}"
consistency_repeats_per_init="{CONSISTENCY_REPEATS_PER_INIT}"

# Build command arguments (tyro.cli uses --args.field-name format)
CMD_ARGS=(
    --args.pretrained-path "${{your_ckpt}}"
    --args.host "${{host}}"
    --args.port "${{base_port}}"
    --args.task-suite-name "${{task_suite_name}}"
    --args.num-trials-per-task "${{num_trials_per_task}}"
    --args.video-out-path "${{video_out_path}}"
    --args.attention-eval-mode "${{attention_eval_mode}}"
    --args.attention-eval-method "${{attention_eval_method}}"
    --args.attention-eval-ratios "${{attention_eval_ratios}}"
    --args.action-noise-std "${{action_noise_std}}"
    --args.consistency-repeats-per-init "${{consistency_repeats_per_init}}"
)

# Add evaluation config path if provided
if [ -n "${{evaluation_config_path}}" ] && [ -f "${{evaluation_config_path}}" ]; then
    CMD_ARGS+=(--args.evaluation-config-path "${{evaluation_config_path}}")
fi

python ./examples/LIBERO/eval_files/eval_libero.py "${{CMD_ARGS[@]}}"
"""
)
client_local.chmod(0o755)
print(f"[Script Generation] Client script: {client_local}", flush=True)
print("[Script Generation] 脚本生成完成", flush=True)

# ------------------------------------------------------------------------------
# 运行与清理
# ------------------------------------------------------------------------------
print("[Execution] 开始执行评估流程...", flush=True)
env_common = os.environ.copy()
env_common.update(
    {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    }
)

server_log = open(LOG_ROOT / "server.log", "w")
server_proc = None
try:
    # Step 1: 启动服务器（starVLA 环境）
    print("[Step 1] 启动策略服务器（starVLA 环境）...", flush=True)
    server_proc = subprocess.Popen(
        ["bash", str(server_local)],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=env_common,
    )
    print(f"[Step 1] Server PID: {server_proc.pid}", flush=True)

    # Step 2: 等待服务器加载
    print(f"[Step 2] 等待 {WAIT_SECS}s 以便服务器就绪...", flush=True)
    time.sleep(WAIT_SECS)
    print(f"[Step 2] 等待完成", flush=True)

    # Step 3: 启动模拟（LIBERO PRO 环境），不拆分 task
    print("[Step 3] 运行 LIBERO PRO 评测（不拆分 task）...", flush=True)
    print(f"[Step 3] Client script path: {client_local}", flush=True)
    print(f"[Step 3] Client log path: {Path(client_log_path) / f'{folder_name}.log'}", flush=True)
    with open(Path(client_log_path) / f"{folder_name}.log", "w") as client_log:
        subprocess.run(
            ["bash", str(client_local)],
            check=True,
            stdout=client_log,
            stderr=subprocess.STDOUT,
            env=env_common,
        )

finally:
    # Step 4: 清理
    print("[Step 4] 清理资源...", flush=True)
    if server_proc and server_proc.poll() is None:
        print("评测完成，停止服务器进程。", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    server_log.close()

print("\n" + "=" * 80)
print("=== LIBERO PRO Evaluation Completed ===")
print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Suite: {TASK_SUITE}")
print(f"Model: {model_type}")
if EVALUATION_CONFIG_PATH:
    print(f"LIBERO PRO config: {EVALUATION_CONFIG_PATH}")
print(f"\nOutput locations:")
print(f"  Client log: {client_log_path}/{folder_name}.log")
print(f"  Videos: {video_out_path}")
print(f"  Server log: {LOG_ROOT}/server.log")
print(f"  Run scripts: {LOG_ROOT}/")
print("=" * 80)