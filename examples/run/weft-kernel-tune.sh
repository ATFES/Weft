#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <sg2044|k1|v100> <kernel|format:phase> <repetitions>" >&2
  exit 2
fi

target=$1
kernel=$2
repetitions=$3
run_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner=${WEFT_TUNE_RUNNER:-kernel}

run_candidate() {
  local meta=$1
  local lmul=$2
  local unroll=$3
  local pipeline=$4
  case "${runner}" in
    kernel)
      WEFT_META_BINDINGS="${meta}" \
      WEFT_AUTO_LMUL_EIGHTHS="${lmul}" \
      WEFT_AUTO_UNROLL="${unroll}" \
      WEFT_AUTO_PIPELINE_DEPTH="${pipeline}" \
        "${run_dir}/weft-kernel.sh" "${target}" "${kernel}" "${repetitions}"
      ;;
    mul-mat)
      local format=${kernel%%:*}
      local phase=${kernel#*:}
      if [[ -z ${format} || ${phase} == "${kernel}" ||
            ( ${phase} != decode && ${phase} != prefill ) ]]; then
        echo "mul-mat tuning expects <format>:<decode|prefill>" >&2
        return 2
      fi
      WEFT_META_BINDINGS="${meta}" \
      WEFT_AUTO_LMUL_EIGHTHS="${lmul}" \
      WEFT_AUTO_UNROLL="${unroll}" \
      WEFT_AUTO_PIPELINE_DEPTH="${pipeline}" \
        "${run_dir}/weft-mul-mat.sh" "${target}" "${format}" "${phase}" \
        "${repetitions}"
      ;;
    vec-dot)
      WEFT_AUTO_LMUL_EIGHTHS="${lmul}" \
      WEFT_AUTO_UNROLL="${unroll}" \
      WEFT_AUTO_PIPELINE_DEPTH="${pipeline}" \
        "${run_dir}/weft-quantized-vec-dot.sh" "${target}" "${kernel}" \
        "${repetitions}"
      ;;
    row-dequantize)
      WEFT_AUTO_LMUL_EIGHTHS="${lmul}" \
      WEFT_AUTO_UNROLL="${unroll}" \
      WEFT_AUTO_PIPELINE_DEPTH="${pipeline}" \
        "${run_dir}/weft-row-dequantize.sh" "${target}" "${kernel}" \
        "${repetitions}"
      ;;
    *)
      echo "unsupported WEFT_TUNE_RUNNER: ${runner}" >&2
      return 2
      ;;
  esac
}

IFS=',' read -r -a unrolls <<< "${WEFT_TUNE_UNROLLS:-1,2,4}"
IFS=',' read -r -a pipelines <<< "${WEFT_TUNE_PIPELINE_DEPTHS:-1,2}"
IFS=',' read -r -a lmuls <<< "${WEFT_TUNE_LMUL_EIGHTHS:-8,16,32,64}"

meta_configs=("")
if [[ -n ${WEFT_TUNE_META_CHOICES:-} ]]; then
  IFS=';' read -r -a dimensions <<< "${WEFT_TUNE_META_CHOICES}"
  for dimension in "${dimensions[@]}"; do
    name=${dimension%%=*}
    values=${dimension#*=}
    if [[ -z ${name} || ${values} == "${dimension}" ]]; then
      echo "invalid WEFT_TUNE_META_CHOICES dimension: ${dimension}" >&2
      exit 2
    fi
    IFS=',' read -r -a choices <<< "${values}"
    expanded=()
    for existing in "${meta_configs[@]}"; do
      for choice in "${choices[@]}"; do
        if [[ -z ${choice} ]]; then
          echo "empty meta choice for ${name}" >&2
          exit 2
        fi
        if [[ -n ${existing} ]]; then
          expanded+=("${existing};${name}=${choice}")
        else
          expanded+=("${name}=${choice}")
        fi
      done
    done
    meta_configs=("${expanded[@]}")
  done
fi

results=$(mktemp /tmp/weft-tune.XXXXXX)
cleanup() {
  status=$?
  trap - EXIT
  rm -f -- "${results}"
  exit "${status}"
}
trap cleanup EXIT

for meta in "${meta_configs[@]}"; do
  for lmul in "${lmuls[@]}"; do
    for unroll in "${unrolls[@]}"; do
      for pipeline in "${pipelines[@]}"; do
        if ! output=$(run_candidate "${meta}" "${lmul}" "${unroll}" "${pipeline}" 2>&1); then
          printf 'rejected lmul_eighths=%s unroll=%s pipeline_depth=%s meta=%s\n' \
            "${lmul}" "${unroll}" "${pipeline}" "${meta:-none}" >&2
          continue
        fi
        numeric=$(sed -n 's/^numeric=//p' <<< "${output}")
        metric_name=cold_gop_s
        metric=$(sed -n 's/^cold_gop_s=//p' <<< "${output}")
        if [[ -z ${metric} ]]; then
          metric_name=cold_melements_s
          metric=$(sed -n 's/^cold_melements_s=//p' <<< "${output}")
        fi
        if [[ -z ${metric} ]]; then
          metric_name=melements_s
          metric=$(sed -n 's/^melements_s=//p' <<< "${output}")
        fi
        if [[ ${numeric} != bit-exact && ${numeric} != within-tolerance ]] ||
           [[ -z ${metric} ]]; then
          echo "candidate produced no numerically valid measurable result" >&2
          exit 1
        fi
        printf 'lmul_eighths=%s unroll=%s pipeline_depth=%s meta=%s %s=%s\n' \
          "${lmul}" "${unroll}" "${pipeline}" "${meta:-none}" \
          "${metric_name}" "${metric}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${metric}" "${lmul}" "${unroll}" "${pipeline}" "${meta:-none}" \
          "${metric_name}" >> "${results}"
      done
    done
  done
done

winner=$(sort -t $'\t' -k1,1gr "${results}" | head -n 1)
if [[ -z ${winner} ]]; then
  echo "no numerically valid measurable candidate" >&2
  exit 1
fi
IFS=$'\t' read -r metric lmul unroll pipeline meta metric_name <<< "${winner}"
printf 'winner lmul_eighths=%s unroll=%s pipeline_depth=%s meta=%s %s=%s\n' \
  "${lmul}" "${unroll}" "${pipeline}" "${meta}" "${metric_name}" "${metric}"
if [[ ${WEFT_TUNE_APPLY_WINNER:-1} == 1 ]]; then
  [[ ${meta} == none ]] && meta=
  printf 'selected_run\n'
  run_candidate "${meta}" "${lmul}" "${unroll}" "${pipeline}"
fi
