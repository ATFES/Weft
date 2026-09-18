"""Dense family: gemm_f32 and gemv_f32 native JIT cases."""
from __future__ import annotations

import time
from array import array

import weft
from weft.runtime import Buffer

from native_jit_common import (
    ABSOLUTE_TOLERANCE, Stopwatch, argument_plans, require_finite,
)
from native_jit_cases import call_arguments, compile_options

GEMM_N = 4096
GEMM_K = 4096
GEMV_M = 14336
GEMV_K = 4096


def _compile(spec, ctx):
    options = compile_options(spec)
    begin = time.perf_counter_ns()
    compiled = weft.compile(spec.kernel_function(), options=options,
                            toolchain=ctx.toolchain)
    return compiled, (time.perf_counter_ns() - begin) / 1e6


def execute(spec, ctx) -> dict:
    plans = None  # dense buffers are declared directly from the kernel contract
    compiled, compile_ms = _compile(spec, ctx)
    try:
        plans = argument_plans(compiled)
        if spec.symbol == "gemm_f32":
            detail = _run_gemm(spec, ctx, compiled, plans)
        elif spec.symbol == "gemv_f32":
            detail = _run_gemv(spec, ctx, compiled, plans)
        else:
            raise KeyError(f"unexpected dense symbol {spec.symbol}")
        detail["compile_ms"] = compile_ms
        return detail
    finally:
        compiled.close()


def _phase_m(spec) -> int:
    if spec.runner == "mul-mat":
        return 1 if spec.request.endswith("decode") else 128
    return 128


def _run_gemm(spec, ctx, compiled, plans) -> dict:
    m, n, k = _phase_m(spec), GEMM_N, GEMM_K
    lhs = array("f", [1.0 / k]) * (m * k)
    rhs = array("f", [1.0]) * (n * k)
    output = array("f", [0.0]) * (m * n)
    buffers = {
        "X": Buffer(lhs, shape=(m, k)),
        "W": Buffer(rhs, shape=(n, k)),
        "Y": Buffer(output, shape=(m, n)),
    }
    arguments = call_arguments(compiled, buffers)
    cold = Stopwatch()
    cold.measure(lambda: compiled(*arguments))
    pairs = ((value, 1.0) for value in output)
    max_error = require_finite(pairs, ABSOLUTE_TOLERANCE)
    warm = Stopwatch()
    for _ in range(ctx.repeat):
        warm.measure(lambda: compiled(*arguments))
    return {
        "verdict_kind": "numeric", "m": m, "n": n, "k": k,
        "input_policy": "dense-fixed-values",
        "max_absolute_error": max_error, "cold_call_ms": cold.median_ms,
        "warm_median_ms": warm.median_ms, "repeat": ctx.repeat,
    }


def _run_gemv(spec, ctx, compiled, plans) -> dict:
    m, k = GEMV_M, GEMV_K
    weights = array("f", [0.5]) * (m * k)
    activation = array("f", [1.0 / k]) * k
    output = array("f", [0.0]) * m
    buffers = {
        "W": Buffer(weights, shape=(m, k)),
        "X": Buffer(activation, shape=(k,)),
        "Y": Buffer(output, shape=(m,)),
    }
    arguments = call_arguments(compiled, buffers)
    cold = Stopwatch()
    cold.measure(lambda: compiled(*arguments))
    pairs = ((value, 0.5) for value in output)
    max_error = require_finite(pairs, ABSOLUTE_TOLERANCE)
    warm = Stopwatch()
    for _ in range(ctx.repeat):
        warm.measure(lambda: compiled(*arguments))
    return {
        "verdict_kind": "numeric", "m": m, "k": k,
        "input_policy": "dense-fixed-values",
        "max_absolute_error": max_error, "cold_call_ms": cold.median_ms,
        "warm_median_ms": warm.median_ms, "repeat": ctx.repeat,
    }
