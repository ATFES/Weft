#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <sg2044|k1|v100> <f32|f16|q1_0|q4_0|q4_1|q5_0|q5_1|q8_0|q2_k|q3_k|q4_k|q4_k_persistent|q4_k_staged|q5_k|q6_k|iq1_s|iq1_m|iq2_s|iq2_s_staged|iq2_xs|iq2_xs_staged|iq2_xxs|iq2_xxs_staged|iq3_s|iq3_xxs|iq4_nl|iq4_xs|tq1_0|tq2_0|mxfp4|nvfp4> <decode|prefill> <repetitions>" >&2
  exit 2
fi

target=$1
format=$2
phase=$3
repetitions=$4
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
compiler="${project_root}/build/tools/weft-compile/weft-compile"
formats=(f16 q1_0 q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq1_m iq2_s iq2_xs iq2_xxs iq3_s iq3_xxs iq4_nl iq4_xs tq1_0 tq2_0 mxfp4 nvfp4)
format_id=-1
if [[ ${format} == q4_k_persistent || ${format} == q4_k_staged ]]; then
  format_id=9
elif [[ ${format} == iq2_s_staged ]]; then
  format_id=14
elif [[ ${format} == iq2_xs_staged ]]; then
  format_id=15
elif [[ ${format} == iq2_xxs_staged ]]; then
  format_id=16
elif [[ ${format} != f32 ]]; then
  for index in "${!formats[@]}"; do
    if [[ ${formats[index]} == "${format}" ]]; then
      format_id=${index}
      break
    fi
  done
fi
if [[ ${format} != f32 && ${format_id} -lt 0 ]]; then
  echo "unsupported MUL_MAT format: ${format}" >&2
  exit 2
fi
if [[ ${format} == q4_k && ${phase} == prefill ]]; then
  format=q4_k_staged
elif [[ ${format} == iq2_xxs && ${phase} == prefill ]]; then
  format=iq2_xxs_staged
elif [[ ${format} == iq2_xs && ${phase} == prefill ]]; then
  format=iq2_xs_staged
elif [[ ${format} == iq2_s && ${phase} == prefill ]]; then
  format=iq2_s_staged
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
    remote_build=/home/zhy001/rvv-v100-llama/build-v100-clang18
    remote_cc=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/bin/clang-18
    remote_cxx=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/bin/clang++-18
    remote_cpu=3
    march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
    vlen=256
    extra_flags="-fno-integrated-as"
    link_path=/home/zhy001/llvm-toolset-18/opt/openEuler/llvm-toolset-18/root/usr/lib64
    runtime_target_define=-DWEFT_TARGET_V100=1
    openmp_library=-lomp
    ;;
  *)
    echo "unsupported Weft target: ${target}" >&2
    exit 2
    ;;
esac

local_root=$(mktemp -d /tmp/weft-mul-mat.XXXXXX)
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

meta=()
physical=()
set_physical_option() {
  local name=$1
  local value=$2
  local retained=()
  local option
  for option in "${physical[@]}"; do
    [[ ${option} == "--${name}="* ]] || retained+=("${option}")
  done
  physical=("${retained[@]}" "--${name}=${value}")
}
runtime_kernel_define=
if [[ ${format} == f32 ]]; then
  dsl=examples/kernels/dense/gemm_f32.py
  kernel=gemm_f32
  runtime=examples/repro/weft/gemm_runtime.cpp
  if [[ ${phase} == prefill ]]; then
    meta=(--meta NC=16 --meta MC=64 --meta NR=4 --meta MR=4)
    physical=(--auto-lmul-eighths=8)
  elif [[ ${target} == sg2044 ]]; then
    meta=(--meta NC=16 --meta MC=64 --meta NR=2 --meta MR=2)
    physical=(--auto-lmul-eighths=16)
  else
    meta=(--meta NC=64 --meta MC=64 --meta NR=1 --meta MR=2)
    physical=(--auto-lmul-eighths=32)
  fi
else
  if [[ ${format} == f16 ]]; then
    kernel=production_mul_mat_f16
    if [[ ${target} == sg2044 ]]; then
      meta=(--meta NC=32 --meta KC=4096 --meta MC=16 --meta MR=4 --meta NR=2 --meta KB=4096)
      physical=(--auto-lmul-eighths=8)
    else
      meta=(--meta NC=32 --meta KC=4096 --meta MC=16 --meta MR=2 --meta NR=2 --meta KB=4096)
      physical=(--auto-lmul-eighths=16)
    fi
  elif [[ ${format} == q4_k_persistent ]]; then
    runtime_kernel_define=-DWEFT_Q4_DERIVED=1
    kernel=production_mul_mat_q4_k_persistent
    physical=(--auto-lmul-eighths=32 --auto-unroll=2 --auto-pipeline-depth=1)
  elif [[ ${format} == q4_k_staged ]]; then
    runtime_kernel_define=-DWEFT_Q4_STAGED=1
    kernel=production_mul_mat_q4_k_staged
    if [[ ${target} == sg2044 ]]; then
      meta=(--meta NC=64 --meta KC=256 --meta MC=16 --meta MR=1)
      physical=(--auto-lmul-eighths=32 --auto-unroll=2 --auto-pipeline-depth=1)
    else
      meta=(--meta NC=64 --meta KC=512 --meta MC=8 --meta MR=2)
      physical=(--auto-lmul-eighths=16 --auto-unroll=2 --auto-pipeline-depth=1)
    fi
  elif [[ ${format} == iq2_xxs_staged ]]; then
    runtime_kernel_define=-DWEFT_IQ2_XXS_STAGED=1
    kernel=production_mul_mat_iq2_xxs_staged
    if [[ ${target} == sg2044 ]]; then
      meta=(--meta NC=64 --meta KC=256 --meta MC=16 --meta MR=1 --meta NR=2)
      physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
    else
      meta=(--meta NC=64 --meta KC=512 --meta MC=8 --meta MR=1 --meta NR=2)
      physical=(--auto-lmul-eighths=8 --auto-unroll=1 --auto-pipeline-depth=1)
    fi
  elif [[ ${format} == iq2_xs_staged ]]; then
    runtime_kernel_define=-DWEFT_IQ2_XS_STAGED=1
    kernel=production_mul_mat_iq2_xs_staged
    if [[ ${target} == sg2044 ]]; then
      meta=(--meta NC=64 --meta KC=256 --meta MC=16 --meta MR=2 --meta NR=2)
      physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
    else
      meta=(--meta NC=64 --meta KC=512 --meta MC=8 --meta MR=4 --meta NR=2)
      physical=(--auto-lmul-eighths=8 --auto-unroll=1 --auto-pipeline-depth=1)
    fi
  elif [[ ${format} == iq2_s_staged ]]; then
    runtime_kernel_define=-DWEFT_IQ2_S_STAGED=1
    kernel=production_mul_mat_iq2_s_staged
    if [[ ${target} == sg2044 ]]; then
      meta=(--meta NC=64 --meta KC=256 --meta MC=16 --meta MR=4 --meta NR=2)
      physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
    else
      meta=(--meta NC=512 --meta KC=1024 --meta MC=8 --meta MR=8 --meta NR=3)
      physical=(--auto-lmul-eighths=8 --auto-unroll=1 --auto-pipeline-depth=1)
    fi
  elif [[ ${format} == iq2_s ]]; then
    kernel=production_mul_mat_iq2_s
    if [[ ${target} == sg2044 ]]; then
      physical=(--auto-lmul-eighths=32 --auto-unroll=1 --auto-pipeline-depth=1)
    else
      physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
    fi
  elif [[ ${format} == iq2_xs ]]; then
    meta=(--meta NR=1)
    iq2_xs_unroll=${WEFT_AUTO_UNROLL:-1}
    if [[ ${target} == sg2044 ]]; then
      physical=(--auto-lmul-eighths=32 --auto-unroll="${iq2_xs_unroll}" --auto-pipeline-depth=1)
    else
      physical=(--auto-lmul-eighths=32 --auto-unroll="${iq2_xs_unroll}" --auto-pipeline-depth=1)
    fi
    kernel=production_mul_mat_iq2_xs
  elif [[ ${format} == q1_0 ]]; then
    if [[ ${phase} == decode ]]; then
      if [[ ${target} == sg2044 ]]; then
        physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
      else
        physical=(--auto-lmul-eighths=8 --auto-unroll=4 --auto-pipeline-depth=1)
      fi
      kernel=production_mul_mat_q1_0_decode
      runtime_kernel_define=-DWEFT_Q10_DECODE=1
    else
      physical=(--auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q1_0
      meta=(--meta NC=32 --meta MC=16 --meta MR=2 --meta NR=4)
    fi
  elif [[ ${format} == q4_0 ]]; then
    if [[ ${phase} == decode ]]; then
      physical=(--auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q4_0_decode
      runtime_kernel_define=-DWEFT_Q40_DECODE=1
    else
      physical=(--auto-unroll=1 --auto-pipeline-depth=2)
      kernel=production_mul_mat_q4_0
      if [[ ${target} == sg2044 ]]; then
        meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=1)
      else
        meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=2)
      fi
    fi
  elif [[ ${format} == q5_0 ]]; then
    if [[ ${phase} == decode ]]; then
      if [[ ${target} == sg2044 ]]; then
        physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
      else
        physical=(--auto-lmul-eighths=8 --auto-unroll=1 --auto-pipeline-depth=1)
      fi
      kernel=production_mul_mat_q5_0_decode
      runtime_kernel_define=-DWEFT_Q50_DECODE=1
    else
      physical=(--auto-lmul-eighths=8 --auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q5_0
      meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=2)
    fi
  elif [[ ${format} == q5_1 ]]; then
    if [[ ${phase} == decode ]]; then
      if [[ ${target} == sg2044 ]]; then
        physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
      else
        physical=(--auto-lmul-eighths=8 --auto-unroll=1 --auto-pipeline-depth=1)
      fi
      kernel=production_mul_mat_q5_1_decode
      runtime_kernel_define=-DWEFT_Q51_DECODE=1
    else
      kernel=production_mul_mat_q5_1
      if [[ ${target} == sg2044 ]]; then
        physical=(--auto-unroll=1 --auto-pipeline-depth=1)
        meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=1)
      else
        physical=(--auto-unroll=1 --auto-pipeline-depth=1)
        meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=2)
      fi
    fi
  elif [[ ${format} == q8_0 ]]; then
    if [[ ${phase} == decode ]]; then
      physical=(--auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q8_0_decode
      runtime_kernel_define=-DWEFT_Q80_DECODE=1
    else
      physical=(--auto-unroll=1 --auto-pipeline-depth=2)
      kernel=production_mul_mat_q8_0
      meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=2)
    fi
  elif [[ ${format} == tq2_0 ]]; then
    if [[ ${phase} == decode ]]; then
      physical=(--auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_tq2_0_decode
      runtime_kernel_define=-DWEFT_TQ20_DECODE=1
    else
      physical=(--auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_tq2_0
      meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=2)
    fi
  elif [[ ${format} == q2_k ]]; then
    if [[ ${phase} == decode ]]; then
      physical=(--auto-unroll=8 --auto-pipeline-depth=1)
      if [[ ${target} == sg2044 ]]; then
        physical+=(--auto-scalar-load-prime=1)
      fi
      runtime_kernel_define=-DWEFT_Q2K_DECODE=1
      kernel=production_mul_mat_q2_k_decode
    else
      kernel=production_mul_mat_q2_k
      meta=(--meta NC=32 --meta MC=32 --meta MR=1 --meta NR=2)
      physical=(--auto-lmul-eighths=8 --auto-unroll=4 --auto-pipeline-depth=1)
    fi
  elif [[ ${format} == q3_k ]]; then
    if [[ ${phase} == decode ]]; then
      if [[ ${target} == sg2044 ]]; then
        physical=(--auto-lmul-eighths=32 --auto-unroll=1 --auto-pipeline-depth=1 --auto-scalar-load-prime=1)
      else
        physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
      fi
      kernel=production_mul_mat_q3_k_decode
      runtime_kernel_define=-DWEFT_Q3K_DECODE=1
    else
      physical=(--auto-lmul-eighths=32 --auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q3_k
      if [[ ${target} == sg2044 ]]; then
        meta=(--meta NC=32 --meta MC=2 --meta MR=1 --meta NR=1)
      else
        meta=(--meta NC=32 --meta MC=8 --meta MR=1 --meta NR=1)
      fi
    fi
  elif [[ ${format} == q4_k ]]; then
    if [[ ${phase} == decode ]]; then
      physical=(--auto-unroll=8 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q4_k_decode
      runtime_kernel_define=-DWEFT_Q4K_DECODE=1
    else
      kernel=production_mul_mat_q4_k
    fi
  elif [[ ${format} == q5_k ]]; then
    physical=(--auto-unroll=1 --auto-pipeline-depth=1)
    if [[ ${phase} == decode ]]; then
      kernel=production_mul_mat_q5_k_decode
      runtime_kernel_define=-DWEFT_Q5K_DECODE=1
    else
      kernel=production_mul_mat_q5_k
      meta=(--meta NC=32 --meta MC=16 --meta MR=2 --meta NR=4)
    fi
  elif [[ ${format} == q6_k ]]; then
    if [[ ${phase} == decode ]]; then
      physical=(--auto-lmul-eighths=32 --auto-unroll=2 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q6_k_decode
      runtime_kernel_define=-DWEFT_Q6K_DECODE=1
    else
      physical=(--auto-lmul-eighths=32 --auto-unroll=4 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q6_k
      meta=(--meta NC=64 --meta MC=16 --meta MR=4 --meta NR=2)
    fi
  elif [[ ${format} == q4_1 ]]; then
    if [[ ${phase} == decode ]]; then
      physical=(--auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q4_1_decode
      runtime_kernel_define=-DWEFT_Q41_DECODE=1
    else
      physical=(--auto-unroll=1 --auto-pipeline-depth=1)
      kernel=production_mul_mat_q4_1
      if [[ ${target} == sg2044 ]]; then
        meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=1)
      else
        meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=2)
      fi
    fi
  elif [[ ${format} == iq4_nl ]]; then
    physical=(--auto-unroll=1 --auto-pipeline-depth=1)
    if [[ ${phase} == decode ]]; then
      kernel=production_mul_mat_iq4_nl_decode
      runtime_kernel_define=-DWEFT_IQ4_NL_DECODE=1
    else
      kernel=production_mul_mat_iq4_nl
      meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=2)
    fi
  elif [[ ${format} == iq1_s ]]; then
    kernel=production_mul_mat_iq1_s
    if [[ ${phase} == decode ]]; then
      physical=(--auto-lmul-eighths=32 --auto-unroll=4 --auto-pipeline-depth=1)
    else
      physical=(--auto-lmul-eighths=8 --auto-unroll=1 --auto-pipeline-depth=1)
    fi
  elif [[ ${format} == iq2_xxs ]]; then
    if [[ ${target} == sg2044 ]]; then
      physical=(--auto-lmul-eighths=32 --auto-unroll=2 --auto-pipeline-depth=1)
    else
      physical=(--auto-lmul-eighths=32 --auto-unroll=4 --auto-pipeline-depth=1)
    fi
    if [[ ${phase} == decode ]]; then
      kernel=production_mul_mat_iq2_xxs_decode
      runtime_kernel_define=-DWEFT_IQ2_XXS_DECODE=1
    else
      kernel=production_mul_mat_iq2_xxs
    fi
  elif [[ ${format} == iq3_s || ${format} == iq3_xxs ]]; then
    kernel=production_mul_mat_${format}
    if [[ ${target} == sg2044 ]]; then
      physical=(--auto-lmul-eighths=32 --auto-unroll=1 --auto-pipeline-depth=1)
    else
      physical=(--auto-lmul-eighths=16 --auto-unroll=1 --auto-pipeline-depth=1)
    fi
  elif [[ ${format} == iq4_xs ]]; then
    physical=(--auto-unroll=1 --auto-pipeline-depth=1)
    if [[ ${phase} == decode ]]; then
      kernel=production_mul_mat_iq4_xs_decode
      runtime_kernel_define=-DWEFT_IQ4_XS_DECODE=1
    else
      kernel=production_mul_mat_iq4_xs
      meta=(--meta NC=32 --meta MC=16 --meta MR=2 --meta NR=4)
    fi
  elif [[ ${format} == mxfp4 ]]; then
    physical=(--auto-unroll=1 --auto-pipeline-depth=1)
    if [[ ${phase} == decode ]]; then
      kernel=production_mul_mat_mxfp4_decode
      runtime_kernel_define=-DWEFT_MXFP4_DECODE=1
    else
      kernel=production_mul_mat_mxfp4
      meta=(--meta NC=32 --meta MC=16 --meta MR=2 --meta NR=4)
    fi
  elif [[ ${format} == nvfp4 ]]; then
    physical=(--auto-unroll=1 --auto-pipeline-depth=1)
    if [[ ${phase} == decode ]]; then
      kernel=production_mul_mat_nvfp4_decode
      runtime_kernel_define=-DWEFT_NVFP4_DECODE=1
    else
      kernel=production_mul_mat_nvfp4
      meta=(--meta NC=32 --meta MC=16 --meta MR=2 --meta NR=4)
    fi
  elif [[ ${format} == tq1_0 ]]; then
    physical=(--auto-unroll=1 --auto-pipeline-depth=1)
    if [[ ${phase} == decode ]]; then
      kernel=production_mul_mat_tq1_0_decode
      runtime_kernel_define=-DWEFT_TQ10_DECODE=1
    else
      kernel=production_mul_mat_tq1_0
    fi
    if [[ ${target} == sg2044 ]]; then
      physical+=(--auto-lmul-eighths=16)
      if [[ ${phase} == prefill ]]; then
        meta=(--meta NC=32 --meta MC=16 --meta MR=2 --meta NR=1)
      fi
    else
      physical+=(--auto-lmul-eighths=8)
      if [[ ${phase} == prefill ]]; then
        meta=(--meta NC=32 --meta MC=16 --meta MR=4 --meta NR=2)
      fi
    fi
  else
    kernel=production_mul_mat_${format}
  fi
  dsl=examples/kernels/mul_mat/${kernel#production_mul_mat_}.py
  runtime=examples/repro/weft/mul_mat_runtime.cpp
fi
if [[ -n ${WEFT_META_BINDINGS:-} ]]; then
  meta=()
  IFS=';' read -r -a requested_meta <<< "${WEFT_META_BINDINGS}"
  for binding in "${requested_meta[@]}"; do
    [[ -n ${binding} ]] && meta+=(--meta "${binding}")
  done
fi
[[ -n ${WEFT_AUTO_UNROLL:-} ]] &&
  set_physical_option auto-unroll "${WEFT_AUTO_UNROLL}"
[[ -n ${WEFT_AUTO_PIPELINE_DEPTH:-} ]] &&
  set_physical_option auto-pipeline-depth "${WEFT_AUTO_PIPELINE_DEPTH}"
[[ -n ${WEFT_AUTO_LMUL_EIGHTHS:-} ]] &&
  set_physical_option auto-lmul-eighths "${WEFT_AUTO_LMUL_EIGHTHS}"
[[ -n ${WEFT_AUTO_SCALAR_LOAD_PRIME:-} ]] &&
  set_physical_option auto-scalar-load-prime "${WEFT_AUTO_SCALAR_LOAD_PRIME}"
PYTHONPATH="${project_root}/python:${project_root}/examples" python -m weft \
  "${project_root}/${dsl}" > "${local_root}/kernel.mlir"
"${compiler}" "${local_root}/kernel.mlir" --emit=intrinsic-c \
  --march="${march}" --abi=lp64d --vlen-bits="${vlen}" "${meta[@]}" \
  "${physical[@]}" \
  -o "${local_root}/kernel.c"
cp "${project_root}/${runtime}" "${local_root}/runtime.cpp"

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
printf -v phase_argument '%q' "${phase}"
printf -v repetitions_argument '%q' "${repetitions}"
printf -v keep_artifacts_argument '%q' "${WEFT_KEEP_ARTIFACTS:-0}"

tar -C "${local_root}" -cf - kernel.c runtime.cpp |
  ssh "${remote_host}" "
    set -eu
    remote_root=\$(mktemp -d /tmp/weft-mul-mat.XXXXXX)
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
      \${extra_flags} -march=\"\${march}\" -mabi=lp64d -c kernel.c -o kernel.o
    \"\${cxx}\" -O3 -std=c++17 -Wall -Wextra -Werror \
      -Wno-unused-const-variable -ffp-contract=fast \${extra_flags} \
      \${runtime_target_define} ${runtime_kernel_define} \
      -DWEFT_MUL_MAT_FORMAT=\"\${format_id}\" \
      -march=\"\${march}\" -mabi=lp64d \
      -I\"\${source_root}/ggml/include\" -I\"\${source_root}/ggml/src\" \
      -I\"\${source_root}/ggml/src/ggml-cpu\" runtime.cpp kernel.o \
      -L\"\${build_root}/bin\" -L\"\${link_path}\" \
      -Wl,-rpath,\"\${build_root}/bin:\${link_path}\" \
      -Wl,--no-as-needed -lggml -lggml-cpu -lggml-base \${openmp_library} -lm -ldl -pthread \
      -o runtime
    taskset -c \"\${cpu}\" ./runtime ${phase_argument} ${repetitions_argument}
  "
