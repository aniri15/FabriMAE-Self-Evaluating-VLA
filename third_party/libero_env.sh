#!/usr/bin/env bash
# Shared LIBERO environment setup for MAE framework models.
# Benchmark bundle lives in a separate repo, linked at third_party/LIBERO-REFLECT.
#
#   source "${MAIN_ROOT}/third_party/libero_env.sh"
#   setup_libero_env "libero"           # standard LIBERO
#   setup_libero_env "libero_reflect"   # LIBERO-PRO swap / OOD (alias: libero_pro)

if [[ -z "${MAIN_ROOT:-}" ]]; then
  echo "libero_env.sh: MAIN_ROOT is not set" >&2
  return 1 2>/dev/null || exit 1
fi

_libero_reflect_root() {
  local root="${LIBERO_REFLECT_ROOT:-${MAIN_ROOT}/third_party/LIBERO-REFLECT}"
  if [[ ! -e "${root}" ]]; then
    echo "libero_env.sh: LIBERO-REFLECT not found at '${root}'." >&2
    echo "  Clone the LIBERO-REFLECT repo and run: bash third_party/setup_libero_reflect.sh" >&2
    return 1 2>/dev/null || exit 1
  fi
  (cd "${root}" && pwd)
}

setup_libero_env() {
  local benchmark="${1:-libero}"
  case "${benchmark}" in
    libero_pro)
      benchmark="libero_reflect"
      ;;
  esac

  local reflect_root
  reflect_root="$(_libero_reflect_root)" || return 1
  export LIBERO_REFLECT_ROOT="${reflect_root}"

  # shellcheck source=/dev/null
  source "${reflect_root}/libero_env.sh"
  setup_libero_reflect_env "${benchmark}"
}
