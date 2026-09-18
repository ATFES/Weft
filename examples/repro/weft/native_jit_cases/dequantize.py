"""dequantize family: row dequantization cases for every quantized format."""
from __future__ import annotations

import math
import random
import time
from array import array

import weft
from weft.runtime import Buffer

from native_jit_common import (
    ABSOLUTE_TOLERANCE, Stopwatch, argument_plans, require_finite,
)
from native_jit_cases import (
    FORMAT_INFO, FORMAT_ORDER, TABLE_PARAM_NAMES, call_arguments,
    compile_options, table_buffer,
)

DEQUANTIZE_K = 4096
DEQUANTIZE_ROWS = 8


def execute(spec, ctx) -> dict:
    fmt = spec.request
    info = FORMAT_INFO[fmt]
    k = DEQUANTIZE_K

    options = compile_options(spec)
    begin = time.perf_counter_ns()
    compiled = weft.compile(spec.kernel_function(), options=options,
                            toolchain=ctx.toolchain)
    compile_ms = (time.perf_counter_ns() - begin) / 1e6
    try:
        plans = argument_plans(compiled)
        plan = plans["W"]
        record_bytes = info.wqk // plan.record_elements * plan.storage_bytes

        generator = random.Random(FORMAT_ORDER.index(fmt) * 7919 + 13)
        record = None
        probe = array("f", bytes(4 * info.wqk))
        for _ in range(64):
            candidate = array("B", generator.randbytes(record_bytes))
            ctx.ggml.reference_dequantize(info.dequantize, candidate, probe,
                                          info.wqk)
            if all(math.isfinite(value) for value in probe):
                record = candidate
                break
        if record is None:
            raise RuntimeError(f"no finite random record for {fmt}")

        weight_row = record * (k // info.wqk)
        expected = array("f", bytes(4 * k))
        ctx.ggml.reference_dequantize(info.dequantize, weight_row, expected, k)

        buffers = {"W": Buffer(weight_row, shape=(k,),
                               encoding=info.weight_encoding)}
        for name in TABLE_PARAM_NAMES:
            if name in plans:
                buffers[name] = table_buffer(plans[name], ctx.tables, fmt)
        output = array("f", bytes(4 * k))
        buffers["Y"] = Buffer(output, shape=(k,))
        arguments = call_arguments(compiled, buffers)

        cold = Stopwatch()
        cold.measure(lambda: compiled(*arguments))
        max_error = require_finite(zip(output, expected),
                                   ABSOLUTE_TOLERANCE)
        for _ in range(DEQUANTIZE_ROWS - 1):
            compiled(*arguments)
            max_error = max(max_error, require_finite(
                zip(output, expected), ABSOLUTE_TOLERANCE))

        warm = Stopwatch()
        for _ in range(ctx.repeat):
            warm.measure(lambda: [compiled(*arguments)
                                  for _ in range(DEQUANTIZE_ROWS)])
        return {
            "verdict_kind": "numeric", "format": fmt, "k": k,
            "rows": DEQUANTIZE_ROWS,
            "input_policy": "random-valid-record-replicated",
            "max_absolute_error": max_error, "compile_ms": compile_ms,
            "cold_call_ms": cold.median_ms, "warm_median_ms": warm.median_ms,
            "repeat": ctx.repeat,
        }
    finally:
        compiled.close()
