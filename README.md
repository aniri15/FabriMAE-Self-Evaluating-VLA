# Markov Attention Entropy: Self-Evaluation Framework for VLA

This repository implements Markov Attention Entropy (MAE), a white-box self-evaluation framework across heterogeneous VLAs, and evaluates it on LIBERO-Reflect. Benchmark assets are kept in a separate checkout and linked through `third_party/LIBERO-REFLECT`. Evaluation scripts use `third_party/libero_env.sh` to switch between standard (`libero`) and perturbed (`libero_reflect`) rollouts.

## Paper Results at a Glance

Primary self-evaluation results on the four LIBERO-Reflect subsets are shown below. Each cell reports `AUROC / AUPR / FPR@95` in percent, using the paper-reported all-layer, `Top-1` oriented-entropy configuration. AUROC and AUPR are higher-is-better; FPR@95 is lower-is-better. Bold values are best among the three evaluated policies for the corresponding metric and subset.

| Policy | MAE Score | Reflect-Goal | Reflect-Object | Reflect-Spatial | Reflect-10 |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | MAE-D (`Top-1`) | 63.94 / 43.23 / 61.92 | **90.97** / 75.88 / **28.30** | 66.86 / 50.74 / 75.79 | 54.74 / 27.45 / 86.46 |
| OpenVLA-OFT | MAE-D (`Top-1`) | **97.34** / **96.14** / **7.14** | 80.56 / **81.10** / 63.71 | **92.63** / **93.15** / **39.31** | 78.57 / 64.08 / **45.06** |
| QwenPI-Flow | MAE-C (`Top-1`) | 80.57 / 80.01 / 60.12 | 75.94 / 76.48 / 81.18 | 84.80 / 85.46 / 68.53 | **79.52** / **76.24** / 55.60 |

## Repository Layout

```text
.
├── third_party/
│   ├── libero_env.sh
│   └── setup_libero_reflect.sh
├── openvla/
├── openvla_oft/
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
| **MAE-C** | Latent-Refinement VLAs: QwenPI-Flow | Reliable executions retain stronger visual exploration during refinement. |

For the main results, `Top-1` selects the head with the largest
reliability-oriented entropy in each layer after the MAE-D or MAE-C
orientation is applied.

| Model | Supported Self-Evaluation Signals | Attention Metric |
| --- | --- | --- |
| OpenVLA | `consistency`, `output_stats`, `mae-d` | MAE-D |
| OpenVLA-OFT | `mae-d` | MAE-D |
| QwenPI-Flow | `mae-c` | MAE-C |

MAE-D writes `online_MAE-D_scores.jsonl`. MAE-C writes `online_MAE-C_scores.jsonl`. OpenVLA and OpenVLA-OFT do not use MAE-C in self-evaluation configuration; QwenPI-Flow uses MAE-C for online attention evaluation.

## Setup

Requirements:

- Python 3.10 or newer for LIBERO-Reflect.
- Model-specific environments and checkpoints.
- LIBERO benchmark assets, including BDDL and initialization files, installed
  according to the upstream documentation.

```bash
git clone <REPO_URL> mae-self-eval-vla
cd mae-self-eval-vla
export REPO_ROOT="$(pwd)"

git clone <LIBERO_REFLECT_REPO_URL> ../LIBERO-REFLECT
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
- [OpenVLA](https://github.com/openvla/openvla): vision-language-action policy
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft): optimized fine-tuning and inference for OpenVLA
- [starVLA](https://github.com/starVLA/starVLA): QwenPI-Flow policy implementation
