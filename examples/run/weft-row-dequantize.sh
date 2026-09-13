#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <sg2044|k1|v100> <q1_0|q4_0|q4_1|q5_0|q5_1|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k|iq1_s|iq1_m|iq2_s|iq2_xs|iq2_xxs|iq3_s|iq3_xxs|iq4_nl|iq4_xs|tq1_0|tq2_0|mxfp4|nvfp4> <repetitions>" >&2
  exit 2
fi

target=$1
format=$2
repetitions=$3
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
compiler="${project_root}/build/tools/weft-compile/weft-compile"

formats=(q1_0 q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq1_m iq2_s iq2_xs iq2_xxs iq3_s iq3_xxs iq4_nl iq4_xs tq1_0 tq2_0 mxfp4 nvfp4)
format_id=-1
for index in "${!formats[@]}"; do
  if [[ ${formats[index]} == "${format}" ]]; then
    format_id=${index}
    break
  fi
done
if [[ ${format_id} -lt 0 ]]; then
  echo "unsupported row-dequantization format: ${format}" >&2
  exit 2
fi
case "${target}" in
  sg2044)
    remote_host=rvv
    remote_source=/home/ubuntu/llama.cpp-upstream-native
    remote_build=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv
    remote_cc=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
    remote_cxx=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
    remote_cpu=48
    march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
    vlen=128
    extra_flags='--gcc-toolchain=/opt/tcrv-toolchains/gcc-15.2.0 -B/opt/tcrv-toolchains/binutils-2.46.1/bin -fno-integrated-as'
    link_path=/opt/tcrv-toolchains/gcc-15.2.0/lib
    runtime_target_define=
    openmp_library=-lgomp
    ;;
  k1)
    remote_host=k1
    remote_source=/home/bianbu/tcrv-k1-llama
    remote_build=/home/bianbu/tcrv-k1-llama/build-ime
    remote_cc=/usr/bin/clang-18
    remote_cxx=/usr/bin/clang++-18
    remote_cpu=3
    march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba
    vlen=256
    extra_flags=-fno-integrated-as
    link_path=/usr/lib/riscv64-linux-gnu
    runtime_target_define=-DWEFT_TARGET_K1=1
    openmp_library=-lgomp
    ;;
  v100)
    remote_host=rvv-v100
    remote_source=/home/zhy001/rvv-v100-llama
    remote_build=/home/zhy001/rvv-v100-llama/build-v100-clang18-gomp
    remote_cc=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/bin/clang-18
    remote_cxx=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/bin/clang++-18
    remote_cpu=3
    march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
    vlen=256
    extra_flags="-fno-integrated-as"
    link_path=/usr/lib/gcc/riscv64-openEuler-linux/14
    runtime_target_define=-DWEFT_TARGET_V100=1
    openmp_library=-lgomp
    ;;
  *)
    echo "unsupported Weft target: ${target}" >&2
    exit 2
    ;;
esac

configuration_text=$(python3 "${project_root}/examples/run/kernel_configuration.py" \
  row-dequantize "${target} ${format}" --march="${march}" --vlen-bits="${vlen}" \
  --matrix-extension="none")
mapfile -t selected_configuration <<< "${configuration_text}"
dsl=${selected_configuration[0]}
selected_kernel=${selected_configuration[1]}
physical_auto=("${selected_configuration[@]:3}")
printf 'configuration=%s\n' "${selected_configuration[2]}"

local_root=$(mktemp -d /tmp/weft-row-dequantize.XXXXXX)
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

PYTHONPATH="${project_root}/python:${project_root}/examples" python -m weft \
  "${project_root}/${dsl}" \
  > "${local_root}/kernel.mlir"
"${compiler}" "${local_root}/kernel.mlir" --emit=intrinsic-c \
  --march="${march}" --abi=lp64d --vlen-bits="${vlen}" \
  "${physical_auto[@]}" \
  -o "${local_root}/kernel.c"
cp "${project_root}/examples/repro/weft/row_dequantize_runtime.cpp" \
  "${local_root}/runtime.cpp"

printf -v source_argument '%q' "${remote_source}"
printf -v build_argument '%q' "${remote_build}"
printf -v cc_argument '%q' "${remote_cc}"
printf -v cxx_argument '%q' "${remote_cxx}"
printf -v cpu_argument '%q' "${remote_cpu}"
printf -v march_argument '%q' "${march}"
printf -v extra_flags_argument '%q' "${extra_flags}"
printf -v format_argument '%q' "${format_id}"
printf -v link_path_argument '%q' "${link_path}"
printf -v runtime_target_define_argument '%q' "${runtime_target_define}"
printf -v openmp_library_argument '%q' "${openmp_library}"
printf -v repetitions_argument '%q' "${repetitions}"
printf -v keep_artifacts_argument '%q' "${WEFT_KEEP_ARTIFACTS:-0}"

tar -C "${local_root}" -cf - kernel.c runtime.cpp |
  ssh "${remote_host}" "
    set -eu
    remote_root=\$(mktemp -d /tmp/weft-row-dequantize.XXXXXX)
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
    source_root=${source_argument}
    build_root=${build_argument}
    cc=${cc_argument}
    cxx=${cxx_argument}
    compiler_lib_dir=\$(cd -- \"\$(dirname -- \"\${cc}\")/../lib64\" 2>/dev/null && pwd -P || true)
    if [ -d \"\${compiler_lib_dir}\" ]; then
      export LD_LIBRARY_PATH=\"\${compiler_lib_dir}:\${LD_LIBRARY_PATH:-}\"
      export LIBRARY_PATH=\"\${compiler_lib_dir}:\${LIBRARY_PATH:-}\"
    fi
    cpu=${cpu_argument}
    march=${march_argument}
    extra_flags=${extra_flags_argument}
    format_id=${format_argument}
    link_path=${link_path_argument}
    runtime_target_define=${runtime_target_define_argument}
    openmp_library=${openmp_library_argument}

    if [ \"\${keep_artifacts}\" = 1 ]; then
      \"\${cc}\" -O3 -std=c11 -Wall -Wextra -Werror -ffp-contract=fast \
        \${extra_flags} -march=\"\${march}\" -mabi=lp64d -S kernel.c -o kernel.s
    fi

    \"\${cc}\" -O3 -std=c11 -Wall -Wextra -Werror -ffp-contract=fast \
      \${extra_flags} -march=\"\${march}\" -mabi=lp64d \
      -c kernel.c -o kernel.o
    \"\${cxx}\" -O3 -std=c++17 -Wall -Wextra -Werror \
      -Wno-unused-const-variable -ffp-contract=fast \${extra_flags} \
      \${runtime_target_define} \
      -DWEFT_ROW_FORMAT=\"\${format_id}\" \
      -march=\"\${march}\" -mabi=lp64d \
      -I\"\${source_root}/ggml/include\" \
      -I\"\${source_root}/ggml/src\" \
      -I\"\${source_root}/ggml/src/ggml-cpu\" \
      runtime.cpp kernel.o -L\"\${build_root}/bin\" \
      -L\"\${link_path}\" \
      -Wl,-rpath,\"\${build_root}/bin:\${link_path}\" \
      -Wl,--no-as-needed -lggml -lggml-cpu -lggml-base \${openmp_library} -lm -ldl -pthread \
      -o runtime
    taskset -c \"\${cpu}\" ./runtime ${repetitions_argument}
  "
