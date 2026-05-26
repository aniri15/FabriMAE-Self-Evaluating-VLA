#!/usr/bin/env bash
# Link third_party/LIBERO-REFLECT to an external LIBERO-REFLECT git checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REFLECT_SRC="${LIBERO_REFLECT_ROOT:-${MAIN_ROOT}/../LIBERO-REFLECT}"
LINK_PATH="${MAIN_ROOT}/third_party/LIBERO-REFLECT"
LINK_DIR="$(dirname "${LINK_PATH}")"

if [[ -d "${LINK_PATH}" && ! -L "${LINK_PATH}" ]]; then
  echo "Removing old directory ${LINK_PATH} (replace with symlink)."
  rm -rf "${LINK_PATH}"
fi

if [[ ! -e "${REFLECT_SRC}" ]]; then
  echo "Warning: LIBERO-REFLECT not found at ${REFLECT_SRC}" >&2
  echo "  Clone the LIBERO-REFLECT repository, set LIBERO_REFLECT_ROOT if needed, then re-run." >&2
fi

REL_TARGET="$(python3 - <<PY
import os
print(os.path.relpath(os.path.expanduser("${REFLECT_SRC}"), "${LINK_DIR}"))
PY
)"
ln -sfn "${REL_TARGET}" "${LINK_PATH}"

# Model convenience symlinks (keep legacy import/dir names where needed).
ln -sfn ../third_party/LIBERO-REFLECT/standard "${MAIN_ROOT}/openvla_oft/LIBERO"
ln -sfn ../../../../third_party/LIBERO-REFLECT/reflect \
  "${MAIN_ROOT}/openvla/experiments/robot/libero/LIBERO_PRO"
ln -sfn ../../../../../third_party/LIBERO-REFLECT/reflect \
  "${MAIN_ROOT}/openvla_oft/openvla-oft_code/experiments/robot/libero/LIBERO_PRO"

# Optional alias symlink for new naming.
ln -sfn ../third_party/LIBERO-REFLECT/reflect \
  "${MAIN_ROOT}/openvla/experiments/robot/libero/LIBERO_REFLECT" 2>/dev/null || true
ln -sfn ../../../../../third_party/LIBERO-REFLECT/reflect \
  "${MAIN_ROOT}/openvla_oft/openvla-oft_code/experiments/robot/libero/LIBERO_REFLECT" 2>/dev/null || true

# Retire separate LIBERO / LIBERO_PRO links if present.
rm -f "${MAIN_ROOT}/third_party/LIBERO" "${MAIN_ROOT}/third_party/LIBERO_PRO"

echo "Linked ${LINK_PATH} -> $(readlink "${LINK_PATH}")"
echo "LIBERO-REFLECT source: ${REFLECT_SRC}"
echo "Done."
