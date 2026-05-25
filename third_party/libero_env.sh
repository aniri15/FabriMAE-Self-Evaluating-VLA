#!/usr/bin/env bash
# Shared LIBERO / LIBERO-PRO environment setup for all models under main/.
# Source from repo launch scripts:
#   source "${MAIN_ROOT}/third_party/libero_env.sh"
#   setup_libero_env "libero"      # or "libero_pro"
#
# Design notes:
# - Standard LIBERO and LIBERO-PRO use separate install trees under third_party/.
# - LIBERO_CONFIG_PATH points to separate config directories (no ~/.libero bleed).
# - config.yaml is written deterministically before Python starts.
# - Callers prepend their model repo to PYTHONPATH *after* this function returns.

if [[ -z "${MAIN_ROOT:-}" ]]; then
  echo "libero_env.sh: MAIN_ROOT is not set" >&2
  return 1 2>/dev/null || exit 1
fi

_libero_env_write_config() {
  local config_dir="$1"
  local pkg_root="$2"
  mkdir -p "${config_dir}"
  cat > "${config_dir}/config.yaml" <<EOF
benchmark_root: ${pkg_root}
bddl_files: ${pkg_root}/bddl_files
init_states: ${pkg_root}/init_files
datasets: ${pkg_root}/../datasets
assets: ${pkg_root}/assets
EOF
}

setup_libero_env() {
  local benchmark="${1:-libero}"
  local libero_home="${MAIN_ROOT}/third_party/LIBERO"
  local libero_pro_home="${MAIN_ROOT}/third_party/LIBERO_PRO"
  local libero_pkg="${libero_home}/libero/libero"
  local libero_pro_pkg="${libero_pro_home}/libero/libero"

  case "${benchmark}" in
    libero)
      export LIBERO_BENCHMARK="libero"
      export LIBERO_PATH="${libero_home}"
      export LIBERO_HOME="${libero_home}"
      export LIBERO_CONFIG_PATH="${libero_home}/.libero_config"
      _libero_env_write_config "${LIBERO_CONFIG_PATH}" "${libero_pkg}"
      ;;
    libero_pro)
      export LIBERO_BENCHMARK="libero_pro"
      export LIBERO_PATH="${libero_pro_home}"
      export LIBERO_HOME="${libero_pro_home}"
      export LIBERO_CONFIG_PATH="${libero_pro_home}/libero_config"
      _libero_env_write_config "${LIBERO_CONFIG_PATH}" "${libero_pro_pkg}"
      ;;
    *)
      echo "setup_libero_env: unsupported benchmark '${benchmark}' (use libero | libero_pro)" >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac

  export LIBERO_PRO_HOME="${libero_pro_home}"
  export LIBERO_PRO_EVAL_CONFIG="${LIBERO_PRO_EVAL_CONFIG:-${libero_pro_home}/evaluation_config_swap.yaml}"
}
