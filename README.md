# MAE Self-Evaluation Framework for VLA

LIBERO / LIBERO-PRO self-evaluation for three VLA baselines, with shared benchmarks in `third_party/` (`LIBERO`, `LIBERO_PRO`). Launch scripts source `third_party/libero_env.sh` to switch `libero` / `libero_pro` without mixing configs.

**Repository:** [github.com/aniri15/MAE_Self_Evaluation_Framework_for_VLA](https://github.com/aniri15/MAE_Self_Evaluation_Framework_for_VLA)

## Layout

```
.
├── third_party/LIBERO, LIBERO_PRO, libero_env.sh
├── openvla/          # discrete OpenVLA-7B
├── openvla_oft/      # OpenVLA-OFT
└── starVLA/          # Qwen3-VL-PI
```

Symlinks: `openvla/.../LIBERO_PRO` and `openvla_oft/.../LIBERO_PRO` → `third_party/LIBERO_PRO`; `openvla_oft/LIBERO` → `third_party/LIBERO`.

## Self-eval modes (MAE-D / MAE-C)

User-facing metric names: **MAE-D** (top-k visual entropy; internal compute: `mad`) and **MAE-C** (bottom-k visual entropy; internal compute: `mac`).

| Model | Rollout signals | Attention score |
|-------|-----------------|-----------------|
| **OpenVLA** | `consistency`, `output_stats`, **`mae-d`** | **MAE-D only** (online; not MAE-C) |
| **OpenVLA-OFT** | `consistency`, **`mae-d`** | **MAE-D only** (online; not MAE-C) |
| **PI (starVLA)** | action consistency + online attention eval | **MAE-C** (online; default in eval mode) |

**MAE-D (OpenVLA / OFT):** `SELF_EVAL_MODE=mae-d` — extracts attention each query, computes top-k visual entropy per layer (no `.pt` save), writes `online_MAE-D_scores.jsonl`.

**MAE-C (PI):** `--args.attention-eval-mode eval` and `--args.attention-eval-method mae-c`. OpenVLA/OFT reject MAE-C in self-eval config.

## Prerequisites

```bash
git clone git@github.com:aniri15/MAE_Self_Evaluation_Framework_for_VLA.git
cd MAE_Self_Evaluation_Framework_for_VLA

export REPO_ROOT="$(pwd)"
export PRETRAINED_CHECKPOINT=/path/to/checkpoint
export CACHE_ROOT=/path/to/.cache          # optional
export CONDA_ENV=<env>                     # see table below
```

Shell scripts set `MAIN_ROOT` to the repository root automatically; you can also `export MAIN_ROOT="${REPO_ROOT}"` before running eval.

| Model | Conda env (default in scripts) |
|-------|--------------------------------|
| OpenVLA | `openvla-origin` |
| OpenVLA-OFT | `openvla` |
| PI server | `starVLA` |
| PI LIBERO client | your LIBERO env |

Python ≥ 3.10 for LIBERO-PRO. Install LIBERO assets (BDDL, init files) as in upstream docs.

---

## OpenVLA

> **MAE naming:** MAE-D = user-facing metric (`self_eval_mode=mae-d`, output `online_MAE-D_scores.jsonl`); internal computation uses `mad` (top-k visual entropy). MAE-C is PI-only.

```bash
cd openvla
export CONDA_ENV=openvla-origin
export PRETRAINED_CHECKPOINT=...

# Rollout self-eval (pick one mode)
bash openvla_run_libero_self_eval.sh libero libero_spatial consistency
bash openvla_run_libero_self_eval.sh libero libero_spatial output_stats
bash openvla_run_libero_self_eval.sh libero libero_spatial mae-d
```

Output: `eval_results/<libero|libero_pro>/<ckpt>/<suite>/<date>/<run>/` — `self_eval_{consistency|output_stats}_scores.json` or `online_MAE-D_scores.jsonl`.

---

## OpenVLA-OFT

> **MAE naming:** MAE-D = user-facing metric (`self_eval_mode=mae-d`, output `online_MAE-D_scores.jsonl`); internal computation uses `mad`. MAE-C is PI-only.

```bash
cd openvla_oft
export CONDA_ENV=openvla
export PRETRAINED_CHECKPOINT=...

bash openvla_oft_run_libero_self_eval.sh libero libero_spatial consistency
bash openvla_oft_run_libero_self_eval.sh libero libero_spatial mae-d
```

Optional: `NUM_TRIALS_PER_TASK`, `CONSISTENCY_REPEATS`, `ACTION_NOISE_STD`, `ATTENTION_EVAL_RATIOS`.

Output: `eval_results/.../self_eval_consistency_scores.json` or `online_MAE-D_scores.jsonl`.

---

## StarVLA / PI

> **MAE naming:** MAE-C = user-facing metric (`attention_eval_method=mae-c`, output `online_MAE-C_scores.jsonl`); internal computation uses `mac` (bottom-k visual entropy). MAE-D is OpenVLA/OFT-only.

Two processes: policy server + LIBERO client.

```bash
# Terminal A
cd starVLA
export CONDA_ENV=starVLA
export PRETRAINED_CHECKPOINT=...
bash examples/LIBERO/eval_files/run_policy_server.sh

# Terminal B — MAE-C online
cd starVLA
export LIBERO_BENCHMARK=libero
export PRETRAINED_CHECKPOINT=...
python examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${PRETRAINED_CHECKPOINT}" \
  --args.task-suite-name libero_spatial \
  --args.attention-eval-mode eval \
  --args.attention-eval-method mae-c \
  --args.attention-eval-ratios 0.01
```

LIBERO-PRO batch (MAE-C): `python submit_libero_eval.py --suite libero_spatial --model pi --pert swap --attention-eval-mode eval --attention-eval-method mae-c --slurm`

Outputs: `online_MAE-C_scores.jsonl`, `online_action_consistency_scores.jsonl`; or `saved_attentions/Pi/...` if saving attention.

---

## LIBERO-PRO perturbations

Configs: `third_party/LIBERO_PRO/evaluation_config_swap.yaml` and `generated_configs/eval_config_<suite>_<pert>.yaml` (`env`, `swap`, `object`, `lang`, `task`).

Override: `export EVALUATION_CONFIG_PATH=...`

---

## Quick commands

From the repository root:

```bash
bash openvla/openvla_run_libero_self_eval.sh libero libero_spatial output_stats
bash openvla_oft/openvla_oft_run_libero_self_eval.sh libero libero_spatial mae-d
# PI: starVLA/examples/LIBERO/eval_files/run_policy_server.sh + eval_libero.py (--attention-eval-method mae-c)
```
