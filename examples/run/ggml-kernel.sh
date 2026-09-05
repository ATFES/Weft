#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <sg2044|k1|v100> <mul_mat|vec_dot|quantize|dequantize|forward|unfamiliar|ime1> <kernel> <repetitions>" >&2
  exit 2
fi

target=$1
family=$2
kernel=$3
repetitions=$4
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

case "${target}" in
  sg2044)
    remote_host=rvv
    remote_source=/home/ubuntu/llama.cpp-upstream-native
    remote_build=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv
    remote_cxx=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
    remote_cpu=48
    remote_march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
    remote_extra_cxx_flags='--gcc-toolchain=/opt/tcrv-toolchains/gcc-15.2.0 -B/opt/tcrv-toolchains/binutils-2.46.1/bin -fno-integrated-as'
    remote_link_path=/opt/tcrv-toolchains/gcc-15.2.0/lib
    remote_extra_define=
    remote_openmp_library=-lgomp
    ;;
  k1)
    remote_host=k1
    remote_source=/home/bianbu/tcrv-k1-llama
    remote_build=/home/bianbu/tcrv-k1-llama/build-ime
    remote_cxx=/usr/bin/clang++-18
    remote_cpu=3
    remote_march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba
    remote_extra_cxx_flags=-fno-integrated-as
    remote_link_path=/usr/lib/riscv64-linux-gnu
    remote_extra_define=
    remote_openmp_library=-lgomp
    ;;
  v100)
    remote_host=rvv-v100
    remote_source=/home/zhy001/rvv-v100-llama
    remote_build=/home/zhy001/rvv-v100-llama/build-v100-clang18
    remote_cxx=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/bin/clang++-18
    remote_cpu=3
    remote_march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
    remote_extra_cxx_flags=-fno-integrated-as
    remote_link_path=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/lib64
    remote_extra_define=
    remote_openmp_library=-lomp
    ;;
  *)
    echo "unsupported GGML target: ${target}" >&2
    exit 2
    ;;
esac

case "${family}" in
  mul_mat)
    runtime=examples/repro/ggml/mul_mat_runtime.cpp
    ;;
  vec_dot)
    runtime=examples/repro/ggml/vec_dot_runtime.cpp
    ;;
  quantize)
    runtime=examples/repro/ggml/quantize_runtime.cpp
    ;;
  dequantize)
    runtime=examples/repro/ggml/dequantize_runtime.cpp
    ;;
  forward)
    runtime=examples/repro/ggml/forward_runtime.cpp
    if [[ ${target} == k1 ]]; then
      remote_extra_define=-DGGML_BASELINE_K1=1
    fi
    ;;
  unfamiliar)
    if [[ ${target} != sg2044 ]]; then
      echo "unfamiliar public-op baselines are only recorded on sg2044" >&2
      exit 2
    fi
    runtime=examples/repro/ggml/unfamiliar_runtime.cpp
    remote_cpu=8
    ;;
  ime1)
    if [[ ${target} != k1 ]]; then
      echo "IME1 is only available on target k1" >&2
      exit 2
    fi
    runtime=examples/repro/ggml/ime1_runtime.cpp
    ;;
  *)
    echo "unsupported GGML kernel family: ${family}" >&2
    exit 2
    ;;
esac

local_root=$(mktemp -d /tmp/ggml-kernel.XXXXXX)
cleanup_local() {
  status=$?
  trap - EXIT
  if ! find "${local_root}" -depth -delete; then
    echo "failed to remove local temporary directory: ${local_root}" >&2
    if [[ ${status} -eq 0 ]]; then
      status=1
    fi
  fi
  exit "${status}"
}
trap cleanup_local EXIT

cp "${project_root}/${runtime}" "${local_root}/runtime.cpp"
printf -v kernel_argument '%q' "${kernel}"
printf -v repetitions_argument '%q' "${repetitions}"
printf -v source_argument '%q' "${remote_source}"
printf -v build_argument '%q' "${remote_build}"
printf -v cxx_argument '%q' "${remote_cxx}"
printf -v cpu_argument '%q' "${remote_cpu}"
printf -v march_argument '%q' "${remote_march}"
printf -v extra_cxx_flags_argument '%q' "${remote_extra_cxx_flags}"
printf -v link_path_argument '%q' "${remote_link_path}"
printf -v extra_define_argument '%q' "${remote_extra_define}"
printf -v openmp_library_argument '%q' "${remote_openmp_library}"

tar -C "${local_root}" -cf - runtime.cpp |
  ssh "${remote_host}" "
    set -eu
    remote_root=\$(mktemp -d /tmp/ggml-kernel.XXXXXX)
    cleanup() {
      exit_status=\$?
      trap - EXIT
      if ! find \"\${remote_root}\" -depth -delete; then
        echo \"failed to remove remote temporary directory: \${remote_root}\" >&2
        if [ \"\${exit_status}\" -eq 0 ]; then
          exit_status=1
        fi
      fi
      exit \"\${exit_status}\"
    }
    trap cleanup EXIT
    tar -C \"\${remote_root}\" -xf -
    cd \"\${remote_root}\"

    source_root=${source_argument}
    build_root=${build_argument}
    cxx=${cxx_argument}
    compiler_lib_dir=\$(cd -- \"\$(dirname -- \"\${cxx}\")/../lib64\" 2>/dev/null && pwd -P || true)
    if [ -d \"\${compiler_lib_dir}\" ]; then
      export LD_LIBRARY_PATH=\"\${compiler_lib_dir}:\${LD_LIBRARY_PATH:-}\"
      export LIBRARY_PATH=\"\${compiler_lib_dir}:\${LIBRARY_PATH:-}\"
    fi
    cpu=${cpu_argument}
    march=${march_argument}
    extra_cxx_flags=${extra_cxx_flags_argument}
    link_path=${link_path_argument}
    extra_define=${extra_define_argument}
    openmp_library=${openmp_library_argument}

    \"\${cxx}\" -O3 -std=c++17 -Wall -Wextra -Werror -ffp-contract=fast \
      \${extra_cxx_flags} \${extra_define} \
      -march=\"\${march}\" -mabi=lp64d \
      -I\"\${source_root}/ggml/include\" \
      -I\"\${source_root}/ggml/src\" \
      -I\"\${source_root}/ggml/src/ggml-cpu\" \
      -I\"\${source_root}/ggml/src/ggml-cpu/spacemit\" \
      runtime.cpp -L\"\${build_root}/bin\" \
      -L"\${link_path}" \
      -Wl,-rpath,\"\${build_root}/bin:\${link_path}\" \
      -Wl,--no-as-needed \
      -lggml -lggml-cpu -lggml-base \${openmp_library} -lm -ldl -pthread \
      -o ggml_kernel_runtime

    taskset -c \"\${cpu}\" ./ggml_kernel_runtime ${kernel_argument} ${repetitions_argument}
  "
