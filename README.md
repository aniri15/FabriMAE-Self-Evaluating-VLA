# Markov Attention Entropy: Self-Evaluation Framework for VLA

This repository implements Markov Attention Entropy (MAE), a white-box self-evaluation framework across heterogeneous VLAs, and evaluates it on LIBERO-Reflect. Benchmark assets are kept in a separate checkout and linked through `third_party/LIBERO-REFLECT`. Evaluation scripts use `third_party/libero_env.sh` to switch between standard (`libero`) and perturbed (`libero_reflect`) rollouts.


## Repository Layout

```text
.
├── third_party/
│   ├── libero_env.sh
│   └── setup_libero_reflect.sh
├── openvla/
├── openvla_oft/
├── openpi/
└── starVLA/
```

The `LIBERO-REFLECT` checkout is not stored in this repository. It provides the `standard/` and `reflect/` benchmark trees used by the evaluation scripts.
Convenience symlinks in the model directories point to the checkout under `third_party/LIBERO-REFLECT`.

## Metrics and Supported Modes

MAE summarizes visual attention entropy during latent action generation.
Visual entropy is high when an action state addresses visual patches broadly and low when visual addressing is concentrated. The paper uses two architecture-aware orientations:

| MAE Score | Applied To | Reliability Interpretation |
| --- | --- | --- |
| **MAE-D** | Latent-Readout VLAs: OpenVLA, OpenVLA-OFT | Reliable executions concentrate visual addressing. |
| **MAE-C** | Latent-Refinement VLAs: QwenPI-Flow,PI-0.5 | Reliable executions retain stronger visual exploration during refinement. |

For the main results, `Top-1` selects the head with the largest
reliability-oriented entropy in each layer after the MAE-D or MAE-C
orientation is applied.

| Model | Supported Self-Evaluation Signals | Attention Metric |
| --- | --- | --- |
| OpenVLA | `consistency`, `output_stats`, `mae-d` | MAE-D |
| OpenVLA-OFT | `mae-d` | MAE-D |
| QwenPI-Flow | `mae-c` | MAE-C |
| PI-0.5 | `mae-c` | MAE-C |

MAE-D writes `online_MAE-D_scores.jsonl`. MAE-C writes `online_MAE-C_scores.jsonl`. OpenVLA and OpenVLA-OFT do not use MAE-C in self-evaluation configuration; QwenPI-Flow and PI-0.5 use MAE-C for online attention evaluation.

## Setup

Requirements:

- Python 3.10 or newer for LIBERO-Reflect.
- Model-specific environments and checkpoints.
- LIBERO benchmark assets, including BDDL and initialization files, installed
  according to the upstream documentation.

```bash
git clone https://github.com/aniri15/FabriMAE-Self-Evaluating-VLA.git mae-self-eval-vla
cd mae-self-eval-vla
export REPO_ROOT="$(pwd)"

git clone https://github.com/aniri15/LIBERO-REFLECT.git ../LIBERO-REFLECT
export LIBERO_REFLECT_ROOT="../LIBERO-REFLECT"
bash third_party/setup_libero_reflect.sh

export CACHE_ROOT="${REPO_ROOT}/.cache"  # optional
```

The launch scripts determine the repository root automatically. `REPO_ROOT` is useful when selecting checkpoints in commands below.

| Model | Default Conda Environment in Scripts |
| --- | --- |
| OpenVLA | `openvla-origin` |
| OpenVLA-OFT | `openvla` |
| QwenPI-Flow policy server | `starVLA` |
| QwenPI-Flow LIBERO client | LIBERO evaluation environment |
| PI0.5 | `openpi` |

## Run Evaluations

### OpenVLA

```bash
cd "${REPO_ROOT}/openvla"
export CONDA_ENV=openvla-origin
export PRETRAINED_CHECKPOINT="${REPO_ROOT}/checkpoints/<openvla-checkpoint>"

# Choose one self-evaluation signal.
bash openvla_run_libero_self_eval.sh libero libero_spatial consistency
bash openvla_run_libero_self_eval.sh libero libero_spatial output_stats
bash openvla_run_libero_self_eval.sh libero libero_spatial mae-d
```

Results are written under:

```text
eval_results/<libero|libero_reflect>/<checkpoint>/<suite>/<date>/<run>/
```

The result file is either `self_eval_consistency_scores.json`, `self_eval_output_stats_scores.json`, or `online_MAE-D_scores.jsonl`, depending on the selected signal.

### OpenVLA-OFT

```bash
cd "${REPO_ROOT}/openvla_oft"
export CONDA_ENV=openvla
export PRETRAINED_CHECKPOINT="${REPO_ROOT}/checkpoints/<openvla-oft-checkpoint>"

bash openvla_oft_run_libero_self_eval.sh libero libero_spatial consistency
bash openvla_oft_run_libero_self_eval.sh libero libero_spatial mae-d
```

Optional controls: `NUM_TRIALS_PER_TASK`, `CONSISTENCY_REPEATS`, `ACTION_NOISE_STD`, and `ATTENTION_EVAL_RATIOS`.

Results are written as `self_eval_consistency_scores.json` or `online_MAE-D_scores.jsonl` under the evaluation results directory.

### QwenPI-Flow

QwenPI-Flow is implemented under `starVLA/` and uses two processes: a policy server and a LIBERO client.

```bash
# Terminal A: policy server
cd "${REPO_ROOT}/starVLA"
export CONDA_ENV=starVLA
export PRETRAINED_CHECKPOINT="${REPO_ROOT}/checkpoints/<qwenpi-flow-checkpoint>"
bash examples/LIBERO/eval_files/run_policy_server.sh

# Terminal B: online MAE-C evaluation
cd "${REPO_ROOT}/starVLA"
export LIBERO_BENCHMARK=libero
export PRETRAINED_CHECKPOINT="${REPO_ROOT}/checkpoints/<qwenpi-flow-checkpoint>"
python examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${PRETRAINED_CHECKPOINT}" \
  --args.task-suite-name libero_spatial \
  --args.attention-eval-mode eval \
  --args.attention-eval-method mae-c \
  --args.attention-eval-ratios 0.01
```

To submit a perturbed LIBERO-Reflect run:

```bash
python submit_libero_eval.py \
  --suite libero_spatial \
  --model pi \
  --pert swap \
  --attention-eval-mode eval \
  --attention-eval-method mae-c \
  --slurm
```

Outputs include `online_MAE-C_scores.jsonl` and `online_action_consistency_scores.jsonl`. Saved attention outputs, when enabled, are stored under `saved_attentions/Pi/`.

### FabriX-MAE on pi0.5

FabriX-MAE is the pi0.5 implementation of our MAE-guided verifier-free test-time action selection. It runs on LIBERO-plus through `openpi/` and samples multiple action candidates at inference time.

Environment setup should follow the upstream OpenPI pi0.5 instructions and the LIBERO-plus installation instructions. In practice, this means:

- Install and verify the OpenPI environment for pi0.5 inference.
- Install LIBERO-plus with its assets, BDDL files, and initialization states.
- Download the pi0.5 checkpoints into a cache directory visible to OpenPI.
- Make sure `openpi/scripts/run_pi05_libero_plus_eval.sh` can start the policy server and LIBERO-plus evaluator in the same environment.

Required local layout:

```text
.
├── openpi/
└── <project>_link/
    ├── hf-cache/
    ├── libero/LIBERO-plus/
    └── openpi_storage/
```

The OpenPI scripts infer the repository root and the `*_link` directory automatically. If your layout is different, set:

```bash
export LINK_ROOT="<path_to_link_root>"
export LIBERO_ROOT="${LINK_ROOT}/libero/LIBERO-plus"
export HF_CACHE_ROOT="${LINK_ROOT}/hf-cache"
export STORAGE_ROOT="${LINK_ROOT}/openpi_storage"
```

Common setup:

```bash
cd "${REPO_ROOT}/openpi"

export LINK_ROOT="${LINK_ROOT:-$(find "${REPO_ROOT}" -maxdepth 1 -type d -name '*_link' -print -quit)}"
export STORAGE_ROOT="${STORAGE_ROOT:-${LINK_ROOT}/openpi_storage}"
export LIBERO_ROOT="${LIBERO_ROOT:-${LINK_ROOT}/libero/LIBERO-plus}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-${LINK_ROOT}/hf-cache}"

export MODEL_NAME=pi05_libero
export SUITE=libero_spatial
export PERTURBATION=camera
export TASK_START=0
export NUM_TASKS=10
export EPISODES_PER_TASK=1
export SEED=7
export TTS_NUM_CANDIDATES=10
export SAVE_ATTENTION_METRICS=0
```

If `MAX_STEPS` is not set, `run_pi05_libero_plus_eval.sh` uses ACoT-style per-suite limits:

| Suite | Default max steps |
| --- | ---: |
| `libero_spatial` | 660 |
| `libero_object` | 840 |
| `libero_goal` | 900 |
| `libero_10` | 1560 |

Run FabriX-MAE in independent mode:

```bash
export OUT_ROOT="${STORAGE_ROOT}/data/libero_plus_eval_fabrix_mae_independent"

TTS_MODE=independent \
TTS_SCORE_MODE=mae \
ATTENTION_EVAL_MODE=mac \
ATTENTION_EVAL_RATIOS=0.01 \
bash scripts/run_pi05_libero_plus_eval.sh
```

Independent mode samples `TTS_NUM_CANDIDATES` complete action chunks from the same observation and selects the candidate using the configured score.

Run FabriX-MAE in branch mode:

```bash
export OUT_ROOT="${STORAGE_ROOT}/data/libero_plus_eval_fabrix_mae_branch"

TTS_MODE=branch \
TTS_SCORE_MODE=mae \
ATTENTION_EVAL_MODE=mac \
ATTENTION_EVAL_RATIOS=0.01 \
TTS_NUM_CANDIDATES=10 \
TTS_BRANCH_RATIO=0.7 \
TTS_BRANCH_NOISE_SCALE=0.15 \
SAVE_ATTENTION_METRICS=0 \
bash scripts/run_pi05_libero_plus_eval.sh
```

Branch mode shares the early ODE denoising steps, branches into `TTS_NUM_CANDIDATES` candidates at `TTS_BRANCH_RATIO`, adds noise controlled by `TTS_BRANCH_NOISE_SCALE`, and selects one candidate with the configured MAE/MAC score. In our completed LIBERO-plus fine-grid sweep, the best branch setting was `TTS_BRANCH_RATIO=0.7` and `TTS_BRANCH_NOISE_SCALE=0.15`.

Results are written under:

```text
${OUT_ROOT}/pi05_libero/<suite>/<perturbation>/tasks_0_<N-1>_seed0/
```

Each result directory contains `summary.json`, `episodes.csv`, `gpu_usage.csv`, and `policy_server.log`.

## LIBERO-Reflect Perturbation Configuration

LIBERO-Reflect evaluation configuration files are located at:

```text
third_party/LIBERO-REFLECT/model_configs/evaluation_config_swap.yaml
third_party/LIBERO-REFLECT/reflect/generated_configs/eval_config_<suite>_<pert>.yaml
```

Use a specific configuration with:

```bash
export EVALUATION_CONFIG_PATH="third_party/LIBERO-REFLECT/reflect/generated_configs/<config>.yaml"
```

## Quick Start Commands

From the repository root:

```bash
bash openvla/openvla_run_libero_self_eval.sh libero libero_spatial output_stats
bash openvla/openvla_run_libero_self_eval.sh libero_reflect libero_spatial mae-d
bash openvla_oft/openvla_oft_run_libero_self_eval.sh libero libero_spatial mae-d
```

For QwenPI-Flow, start `starVLA/examples/LIBERO/eval_files/run_policy_server.sh` and
run `eval_libero.py` with `--args.attention-eval-method mae-c`.

## Related Work

This framework builds on the following open-source projects:

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): lifelong robot learning benchmark
- [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO): LIBERO evaluation extension with OOD perturbations
- [LIBERO-PLUS](https://github.com/sylvestf/LIBERO-plus): LIBERO evaluation extension with different perturbations
- [OpenVLA](https://github.com/openvla/openvla): vision-language-action policy
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft): optimized fine-tuning and inference for OpenVLA
- [starVLA](https://github.com/starVLA/starVLA): QwenPI-Flow policy implementation
- [openpi](https://github.com/Physical-Intelligence/openpi): PI implementation
