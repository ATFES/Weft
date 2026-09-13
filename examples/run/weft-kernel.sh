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

matrix_extension=none
runtime_kernel_define=
runtime_phase=
case "${kernel}" in
  q4_k_gemv)
    runtime=examples/repro/weft/q4_k_gemv_runtime.cpp
    ;;
  q4_k_gemv_groups4)
    runtime=examples/repro/weft/q4_k_gemv_runtime.cpp
    runtime_kernel_define=-DWEFT_Q4_GROUPS4=1
    ;;
  ime_i8_contract)
    if [[ ${target} != k1 ]]; then
      echo "ime_i8_contract requires target k1 with IME" >&2
      exit 2
    fi
    runtime=examples/repro/weft/ime_i8_contract_runtime.cpp
    matrix_extension=spacemit-ime1
    ;;
  q8_0_quantize|q8_1_quantize|q8_K_quantize)
    runtime=examples/repro/weft/q8_quantize_runtime.cpp
    case "${kernel}" in
      q8_0_quantize) runtime_kernel_define=-DWEFT_Q8_KIND=0 ;;
      q8_1_quantize) runtime_kernel_define=-DWEFT_Q8_KIND=1 ;;
      q8_K_quantize) runtime_kernel_define=-DWEFT_Q8_KIND=2 ;;
    esac
    ;;
  gemv_f32)
    runtime=examples/repro/weft/gemv_runtime.cpp
    ;;
  gemm_f32)
    runtime=examples/repro/weft/gemm_runtime.cpp
    runtime_phase=prefill
    ;;
  *)
    echo "unsupported Weft kernel: ${kernel}" >&2
    exit 2
    ;;
esac

configuration_text=$(python3 "${project_root}/examples/run/kernel_configuration.py" \
  kernel "${target} ${kernel}" --march="${march}" --vlen-bits="${vlen}" \
  --matrix-extension="${matrix_extension}")
mapfile -t selected_configuration <<< "${configuration_text}"
dsl=${selected_configuration[0]}
selected_kernel=${selected_configuration[1]}
physical_auto=("${selected_configuration[@]:3}")
printf 'configuration=%s\n' "${selected_configuration[2]}"

local_root=$(mktemp -d /tmp/weft-kernel.XXXXXX)
cleanup_local() {
  status=$?
  trap - EXIT
  if [[ ${WEFT_KEEP_ARTIFACTS:-0} == 1 ]]; then
    echo "WEFT_LOCAL_ARTIFACTS=${local_root}" >&2
  else
    find "${local_root}" -depth -delete
  fi
  exit "${status}"
}
trap cleanup_local EXIT

PYTHONPATH="${project_root}/python:${project_root}/examples" python -m weft "${project_root}/${dsl}" \
  > "${local_root}/kernel.mlir"
"${compiler}" "${local_root}/kernel.mlir" --emit=intrinsic-c \
  --march="${march}" --abi=lp64d --vlen-bits="${vlen}" \
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
printf -v keep_artifacts_argument '%q' "${WEFT_KEEP_ARTIFACTS:-0}"

tar -C "${local_root}" -cf - kernel.c runtime.cpp |
  ssh "${remote_host}" "
    set -eu
    remote_root=\$(mktemp -d /tmp/weft-kernel.XXXXXX)
    keep_artifacts=${keep_artifacts_argument}
    cleanup() {
      exit_status=\$?
      trap - EXIT
      if [ \"\${keep_artifacts}\" = 1 ]; then
        echo \"WEFT_REMOTE_ARTIFACTS=${remote_host}:\${remote_root}\" >&2
      else
        find \"\${remote_root}\" -depth -delete
      fi
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
    if [ \"\${keep_artifacts}\" = 1 ]; then
      \"\${cc}\" -O3 -std=c11 -Wall -Wextra -Werror -ffp-contract=fast \
        \${extra_cflags} \
        -march=\"\${march}\" -mabi=lp64d -S kernel.c -o kernel.s
    fi
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
