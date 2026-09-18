#!/usr/bin/env bash
# Native JIT suite entry: weft-jit.sh <target> <case-id|family|all> [repeat]
# Executes the full native JIT runner on the target machine itself.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <v100> <case-id|family|all> [repeat]" >&2
  exit 2
fi

target=$1
selection=$2
repeat=${3:-10}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=targets/v100-native-jit.env
. "${script_dir}/targets/${target}-native-jit.env"

flags=()
for flag in ${WEFT_CFLAGS}; do
  flags+=("--cflag=${flag}")
done

run_local() {
  cd "${WEFT_ROOT}"
  local branch
  branch=$(git branch --show-current)
  if [[ ${branch} != "${WEFT_BRANCH}" ]]; then
    echo "expected branch ${WEFT_BRANCH}, found ${branch}" >&2
    exit 2
  fi
  export LD_LIBRARY_PATH="${LLVM_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  exec taskset -c "${WEFT_CPU}" env PYTHONPATH="${WEFT_PYTHONPATH}" \
    python3 examples/repro/weft/native_jit_runner.py \
    --case "${selection}" \
    --repeat "${repeat}" \
    --compiler "${WEFT_COMPILER}" \
    --cc "${WEFT_CC}" \
    "${flags[@]}"
}

run_remote() {
  local quoted_flags=()
  local flag
  for flag in "${flags[@]}"; do
    quoted_flags+=("$(printf '%q' "${flag}")")
  done
  ssh "${SSH_ALIAS}" "
    set -euo pipefail
    cd '${WEFT_ROOT}'
    branch=\$(git branch --show-current)
    if [ \"\${branch}\" != '${WEFT_BRANCH}' ]; then
      echo \"expected branch ${WEFT_BRANCH}, found \${branch}\" >&2
      exit 2
    fi
    export LD_LIBRARY_PATH='${LLVM_PREFIX}/lib:\${LD_LIBRARY_PATH:-}'
    exec taskset -c '${WEFT_CPU}' env PYTHONPATH='${WEFT_PYTHONPATH}' \\
      python3 examples/repro/weft/native_jit_runner.py \\
      --case '${selection}' --repeat '${repeat}' \\
      --compiler '${WEFT_COMPILER}' --cc '${WEFT_CC}' ${quoted_flags[*]}
  "
}

if [[ "$(uname -m)" == "riscv64" && -d "${WEFT_ROOT}" ]]; then
  run_local
else
  run_remote
fi
