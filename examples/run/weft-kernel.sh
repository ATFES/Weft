#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <sg2044|k1|v100> <q4_k_gemv|q4_k_gemv_groups4|q8_0_quantize|q8_1_quantize|q8_K_quantize|gemv_f32|gemm_f32|ime_i8_contract> <repetitions>" >&2
  exit 2
fi

target=$1
kernel=$2
repetitions=$3
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
compiler="${project_root}/build/tools/weft-compile/weft-compile"

case "${target}" in
  sg2044)
    remote_host=rvv
    remote_cc=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
    remote_cxx=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
    remote_cpu=48
    march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
    vlen=128
    extra_cflags='--gcc-toolchain=/opt/tcrv-toolchains/gcc-15.2.0 -B/opt/tcrv-toolchains/binutils-2.46.1/bin -fno-integrated-as'
    link_path=/opt/tcrv-toolchains/gcc-15.2.0/lib
    runtime_target_define=
    ;;
  k1)
    remote_host=k1
    remote_cc=/usr/bin/clang-18
    remote_cxx=/usr/bin/clang++-18
    remote_cpu=3
    march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba
    vlen=256
    extra_cflags=-fno-integrated-as
    link_path=/usr/lib/riscv64-linux-gnu
    runtime_target_define=-DWEFT_TARGET_K1=1
    ;;
  v100)
    remote_host=rvv-v100
    remote_cc=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/bin/clang-18
    remote_cxx=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/bin/clang++-18
    remote_cpu=3
    march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
    vlen=256
    extra_cflags=-fno-integrated-as
    link_path=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/lib64
    runtime_target_define=-DWEFT_TARGET_V100=1
    ;;
  *)
    echo "unsupported Weft target: ${target}" >&2
    exit 2
    ;;
esac

physical_auto=()
if [[ -n ${WEFT_AUTO_LMUL_EIGHTHS:-} ]]; then
  physical_auto+=(--auto-lmul-eighths "${WEFT_AUTO_LMUL_EIGHTHS}")
fi
if [[ -n ${WEFT_AUTO_UNROLL:-} ]]; then
  physical_auto+=(--auto-unroll "${WEFT_AUTO_UNROLL}")
fi
if [[ -n ${WEFT_AUTO_PIPELINE_DEPTH:-} ]]; then
  physical_auto+=(--auto-pipeline-depth "${WEFT_AUTO_PIPELINE_DEPTH}")
fi
meta=()
matrix_extension=none
runtime_kernel_define=
runtime_phase=
case "${kernel}" in
  q4_k_gemv)
    dsl=examples/kernels/gemv/q4_k.py
    runtime=examples/repro/weft/q4_k_gemv_runtime.cpp
    ;;
  q4_k_gemv_groups4)
    dsl=examples/kernels/gemv/q4_k_groups4.py
    runtime=examples/repro/weft/q4_k_gemv_runtime.cpp
    runtime_kernel_define=-DWEFT_Q4_GROUPS4=1
    if [[ ${target} == sg2044 ]]; then
      [[ -n ${WEFT_AUTO_LMUL_EIGHTHS:-} ]] ||
        physical_auto+=(--auto-lmul-eighths 32)
      [[ -n ${WEFT_AUTO_UNROLL:-} ]] || physical_auto+=(--auto-unroll 2)
      [[ -n ${WEFT_AUTO_PIPELINE_DEPTH:-} ]] ||
        physical_auto+=(--auto-pipeline-depth 1)
    fi
    ;;
  ime_i8_contract)
    if [[ ${target} != k1 ]]; then
      echo "ime_i8_contract requires target k1 with IME" >&2
      exit 2
    fi
    dsl=examples/kernels/dense/ime_i8_contract.py
    runtime=examples/repro/weft/ime_i8_contract_runtime.cpp
    meta=(--meta MR=4 --meta NR=4 --meta KB=8)
    matrix_extension=spacemit-ime1
    ;;
  q8_0_quantize|q8_1_quantize|q8_K_quantize)
    quantize_format=${kernel%_quantize}
    dsl=examples/kernels/quantize/${quantize_format,,}.py
    runtime=examples/repro/weft/q8_quantize_runtime.cpp
    case "${kernel}" in
      q8_0_quantize)
        runtime_kernel_define=-DWEFT_Q8_KIND=0
        if [[ ${target} == sg2044 ]]; then
          [[ -n ${WEFT_AUTO_LMUL_EIGHTHS:-} ]] ||
            physical_auto+=(--auto-lmul-eighths 32)
          [[ -n ${WEFT_AUTO_UNROLL:-} ]] || physical_auto+=(--auto-unroll 1)
          [[ -n ${WEFT_AUTO_PIPELINE_DEPTH:-} ]] ||
            physical_auto+=(--auto-pipeline-depth 1)
        fi
        ;;
      q8_1_quantize)
        runtime_kernel_define=-DWEFT_Q8_KIND=1
        if [[ ${target} == sg2044 ]]; then
          [[ -n ${WEFT_AUTO_LMUL_EIGHTHS:-} ]] ||
            physical_auto+=(--auto-lmul-eighths 64)
        fi
        ;;
      q8_K_quantize)
        runtime_kernel_define=-DWEFT_Q8_KIND=2
        if [[ ${target} == sg2044 ]]; then
          [[ -n ${WEFT_AUTO_LMUL_EIGHTHS:-} ]] ||
            physical_auto+=(--auto-lmul-eighths 16)
          [[ -n ${WEFT_AUTO_UNROLL:-} ]] || physical_auto+=(--auto-unroll 1)
          [[ -n ${WEFT_AUTO_PIPELINE_DEPTH:-} ]] ||
            physical_auto+=(--auto-pipeline-depth 1)
        elif [[ ${target} == k1 ]]; then
          [[ -n ${WEFT_AUTO_LMUL_EIGHTHS:-} ]] ||
            physical_auto+=(--auto-lmul-eighths 16)
        fi
        ;;
    esac
    ;;
  gemv_f32)
    dsl=examples/kernels/dense/gemv_f32.py
    runtime=examples/repro/weft/gemv_runtime.cpp
    meta=(--meta MR=4 --meta KB=64)
    ;;
  gemm_f32)
    dsl=examples/kernels/dense/gemm_f32.py
    runtime=examples/repro/weft/gemm_runtime.cpp
    meta=(--meta MC=64 --meta NC=16 --meta MR=2 --meta NR=2)
    if [[ ${target} == sg2044 && -z ${WEFT_AUTO_LMUL_EIGHTHS:-} ]]; then
      physical_auto+=(--auto-lmul-eighths 16)
    fi
    runtime_phase=prefill
    ;;
  *)
    echo "unsupported Weft kernel: ${kernel}" >&2
    exit 2
    ;;
esac

if [[ -n ${WEFT_META_BINDINGS:-} ]]; then
  meta=()
  IFS=';' read -r -a requested_meta <<< "${WEFT_META_BINDINGS}"
  for binding in "${requested_meta[@]}"; do
    [[ -n ${binding} ]] && meta+=(--meta "${binding}")
  done
fi

local_root=$(mktemp -d /tmp/weft-kernel.XXXXXX)
cleanup_local() {
  status=$?
  trap - EXIT
  find "${local_root}" -depth -delete
  exit "${status}"
}
trap cleanup_local EXIT

PYTHONPATH="${project_root}/python:${project_root}/examples" python -m weft "${project_root}/${dsl}" \
  > "${local_root}/kernel.mlir"
"${compiler}" "${local_root}/kernel.mlir" --emit=intrinsic-c \
  --march="${march}" --abi=lp64d --vlen-bits="${vlen}" "${meta[@]}" \
  "${physical_auto[@]}" \
  --matrix-extension="${matrix_extension}" \
  -o "${local_root}/kernel.c"
cp "${project_root}/${runtime}" "${local_root}/runtime.cpp"

printf -v repetitions_argument '%q' "${repetitions}"
printf -v cc_argument '%q' "${remote_cc}"
printf -v cxx_argument '%q' "${remote_cxx}"
printf -v cpu_argument '%q' "${remote_cpu}"
printf -v march_argument '%q' "${march}"
printf -v extra_cflags_argument '%q' "${extra_cflags}"
printf -v link_path_argument '%q' "${link_path}"
printf -v runtime_target_define_argument '%q' "${runtime_target_define}"
printf -v runtime_kernel_define_argument '%q' "${runtime_kernel_define}"
printf -v runtime_phase_argument '%q' "${runtime_phase}"

tar -C "${local_root}" -cf - kernel.c runtime.cpp |
  ssh "${remote_host}" "
    set -eu
    remote_root=\$(mktemp -d /tmp/weft-kernel.XXXXXX)
    cleanup() {
      exit_status=\$?
      trap - EXIT
      find \"\${remote_root}\" -depth -delete
      exit \"\${exit_status}\"
    }
    trap cleanup EXIT
    tar -C \"\${remote_root}\" -xf -
    cd \"\${remote_root}\"
    cc=${cc_argument}
    cxx=${cxx_argument}
    compiler_lib_dir=\$(cd -- \"\$(dirname -- \"\${cc}\")/../lib64\" 2>/dev/null && pwd -P || true)
    if [ -d \"\${compiler_lib_dir}\" ]; then
      export LD_LIBRARY_PATH=\"\${compiler_lib_dir}:\${LD_LIBRARY_PATH:-}\"
      export LIBRARY_PATH=\"\${compiler_lib_dir}:\${LIBRARY_PATH:-}\"
    fi
    cpu=${cpu_argument}
    march=${march_argument}
    extra_cflags=${extra_cflags_argument}
    link_path=${link_path_argument}
    runtime_target_define=${runtime_target_define_argument}
    runtime_kernel_define=${runtime_kernel_define_argument}
    \"\${cc}\" -O3 -std=c11 -Wall -Wextra -Werror -ffp-contract=fast \
      \${extra_cflags} \
      -march=\"\${march}\" -mabi=lp64d -c kernel.c -o kernel.o
    \"\${cxx}\" -O3 -std=c++17 -Wall -Wextra -Werror -ffp-contract=fast \
      \${extra_cflags} \${runtime_target_define} \${runtime_kernel_define} \
      -march=\"\${march}\" -mabi=lp64d \
      runtime.cpp kernel.o \
      -L\"\${link_path}\" -Wl,-rpath,\"\${link_path}\" -o runtime
    runtime_phase=${runtime_phase_argument}
    if [ -n \"\${runtime_phase}\" ]; then
      taskset -c \"\${cpu}\" ./runtime \"\${runtime_phase}\" ${repetitions_argument}
    else
      taskset -c \"\${cpu}\" ./runtime ${repetitions_argument}
    fi
  "
