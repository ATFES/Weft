"""vec_dot family: one native JIT case per quantized format."""
from __future__ import annotations

import time
from array import array

import weft
from weft.runtime import Buffer

from native_jit_common import (
    ABSOLUTE_TOLERANCE, Stopwatch, argument_plans, require_finite,
)
from native_jit_cases import (
    FORMAT_ORDER, FORMAT_INFO, PARTNER_RECORD_BYTES, TABLE_PARAM_NAMES,
    activation_source, call_arguments, compile_options, table_buffer,
    weight_source,
)

VEC_DOT_K = 4096
VEC_DOT_ROWS = 8


def execute(spec, ctx) -> dict:
    fmt = spec.request
    info = FORMAT_INFO[fmt]
    format_index = FORMAT_ORDER.index(fmt)
    k = VEC_DOT_K

    options = compile_options(spec)
    begin = time.perf_counter_ns()
    compiled = weft.compile(spec.kernel_function(), options=options,
                            toolchain=ctx.toolchain)
    compile_ms = (time.perf_counter_ns() - begin) / 1e6
    try:
        plans = argument_plans(compiled)

        source = weight_source(format_index, info.wqk)
        scratch = bytearray(info.wqk * 4)
        record_bytes = ctx.ggml.quantize_record(
            ctx.tables.ggml_types[fmt], source, scratch)
        plan = plans["W"]
        expected_record = info.wqk // plan.record_elements * plan.storage_bytes
        if record_bytes != expected_record:
            raise AssertionError(
                f"quantized record is {record_bytes} bytes, kernel expects "
                f"{expected_record}")
        weight_row = array("B", scratch[:record_bytes]) * (k // info.wqk)

        activation = activation_source(info.wqk)
        activation_scratch = bytearray(info.wqk * 4)
        ctx.ggml.quantize_row(info.partner, activation, activation_scratch,
                              info.wqk)
        activation_records = info.wqk // info.xqk * PARTNER_RECORD_BYTES[
            info.partner]
        activation_row = array("B", activation_scratch[:activation_records]) \
            * (k // info.xqk)

        expected = ctx.ggml.reference_dot(
            info.vec_dot, k, weight_row, activation_row)

        buffers = {
            "W": Buffer(weight_row, shape=(k,), encoding=info.weight_encoding),
            "X": Buffer(activation_row, shape=(k,),
                        encoding=info.partner_encoding),
        }
        for name in TABLE_PARAM_NAMES:
            if name in plans:
                buffers[name] = table_buffer(plans[name], ctx.tables, fmt)
        output = array("f", [0.0])
        buffers["Y"] = Buffer(output, shape=(1,))
        arguments = call_arguments(compiled, buffers)

        cold = Stopwatch()
        cold.measure(lambda: compiled(*arguments))
        max_error = require_finite(
            ((output[0], expected),), ABSOLUTE_TOLERANCE)
        for _ in range(VEC_DOT_ROWS - 1):
            compiled(*arguments)
            max_error = max(max_error, require_finite(
                ((output[0], expected),), ABSOLUTE_TOLERANCE))

        warm = Stopwatch()
        for _ in range(ctx.repeat):
            warm.measure(lambda: [compiled(*arguments)
                                  for _ in range(VEC_DOT_ROWS)])
        return {
            "verdict_kind": "numeric", "format": fmt, "k": k,
            "rows": VEC_DOT_ROWS,
            "input_policy": "valid-quantized-record-replicated",
            "expected": expected, "max_absolute_error": max_error,
            "compile_ms": compile_ms, "cold_call_ms": cold.median_ms,
            "warm_median_ms": warm.median_ms, "repeat": ctx.repeat,
        }
    finally:
        compiled.close()
