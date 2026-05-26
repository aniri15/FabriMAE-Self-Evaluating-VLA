# Third-party benchmarks (separate git repo)

LIBERO environments are **not** stored in the MAE framework git repo.

They live in the standalone **LIBERO-REFLECT** repository, which contains:

| Subfolder | MAE benchmark mode | Contents |
|-----------|-------------------|----------|
| `standard/` | `libero` | Standard LIBERO |
| `reflect/` | `libero_reflect` | LIBERO-PRO swap / OOD perturbations |

This repo wires that checkout in via `third_party/LIBERO-REFLECT` (symlink).

## Setup

```bash
# 1) Clone LIBERO-REFLECT (any path; sibling ../LIBERO-REFLECT is the default)
git clone <LIBERO-REFLECT_REPO_URL> ../LIBERO-REFLECT

# 2) Wire symlink + model convenience links
bash third_party/setup_libero_reflect.sh
```

If the checkout is elsewhere:

```bash
export LIBERO_REFLECT_ROOT=/path/to/LIBERO-REFLECT
bash third_party/setup_libero_reflect.sh
```

Legacy alias: `LIBERO_BENCHMARK=libero_pro` still maps to `libero_reflect`.

The old `setup_vendor_symlinks.sh` is deprecated; use `setup_libero_reflect.sh`.
