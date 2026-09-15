#!/usr/bin/env bash
# Build the native Weft compiler for the V100 native JIT suite.
# Uses the project default build command (no explicit parallelism).
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=targets/v100-native-jit.env
. "${script_dir}/targets/v100-native-jit.env"

cd "${WEFT_ROOT}"
branch=$(git branch --show-current)
if [[ ${branch} != "${WEFT_BRANCH}" ]]; then
  echo "expected branch ${WEFT_BRANCH}, found ${branch}" >&2
  exit 2
fi

LLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm"
MLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"

cmake -S "${WEFT_ROOT}" -B "${WEFT_BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}"

cmake --build "${WEFT_BUILD_DIR}" --target weft-compile

test -x "${WEFT_COMPILER}"
"${WEFT_COMPILER}" --query-native-target
