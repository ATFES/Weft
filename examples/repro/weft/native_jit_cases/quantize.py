"""quantize family: activation quantizers validated bit-exactly.

The scalar reference mirrors q8_quantize_runtime.cpp: nearbyint (round to
nearest even) with saturation to int8, float32 arithmetic and fp16 scale
storage.  bit-exactness is the correctness contract for encoded output.
"""
from __future__ import annotations

import struct
import time
from array import array

import weft
from weft.runtime import Buffer

from native_jit_common import Stopwatch, argument_plans
from native_jit_cases import call_arguments, compile_options

QUANTIZE_K = 14336
QUANTIZE_ROWS = 4

_KIND_BY_REQUEST = {
    "q8_0_quantize": "q8_0", "q8_1_quantize": "q8_1", "q8_K_quantize": "q8_K",
}
_ENCODING = {"q8_0": "Q8_0", "q8_1": "Q8_1", "q8_K": "Q8_K"}


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _clamp_i8(value: float) -> int:
    rounded = round(value)
    return max(-128, min(127, rounded))


def _f16_bytes(value: float) -> bytes:
    return struct.pack("<e", value)


def reference_q8_0(source: array) -> bytes:
    out = bytearray()
    for base in range(0, len(source), 32):
        block = source[base:base + 32]
        amax = 0.0
        for value in block:
            amax = max(amax, abs(value))
        scale = _f32(amax / 127.0)
        inverse = 0.0 if amax == 0.0 else _f32(1.0 / scale)
        out += _f16_bytes(scale)
        for value in block:
            out.append(_clamp_i8(_f32(value * inverse)) & 0xFF)
    return bytes(out)


def reference_q8_1(source: array) -> bytes:
    out = bytearray()
    for base in range(0, len(source), 32):
        block = source[base:base + 32]
        amax = 0.0
        for value in block:
            amax = max(amax, abs(value))
        scale = _f32(amax / 127.0)
        inverse = 0.0 if amax == 0.0 else _f32(1.0 / scale)
        codes = [_clamp_i8(_f32(value * inverse)) for value in block]
        out += _f16_bytes(scale)
        out += _f16_bytes(_f32(sum(codes) * scale))
        for code in codes:
            out.append(code & 0xFF)
    return bytes(out)


def reference_q8_k(source: array) -> bytes:
    out = bytearray()
    for base in range(0, len(source), 256):
        block = source[base:base + 256]
        maximum = -math_inf
        minimum = math_inf
        for value in block:
            maximum = max(maximum, value)
            minimum = min(minimum, value)
        extreme = maximum if abs(maximum) > abs(minimum) else minimum
        inverse = 0.0 if extreme == 0.0 else _f32(-127.0 / extreme)
        scale = 0.0 if inverse == 0.0 else _f32(1.0 / inverse)
        out += struct.pack("<f", scale)
        group_codes = []
        for group in range(16):
            codes = [_clamp_i8(_f32(source[base + group * 16 + element]
                                    * inverse))
                     for element in range(16)]
            group_codes.append(codes)
            for code in codes:
                out.append(code & 0xFF)
        for codes in group_codes:
            out += struct.pack("<h", sum(codes))
    return bytes(out)


math_inf = float("inf")

_REFERENCES = {
    "q8_0": reference_q8_0, "q8_1": reference_q8_1, "q8_K": reference_q8_k,
}


def execute(spec, ctx) -> dict:
    kind = _KIND_BY_REQUEST[spec.request]
    k = QUANTIZE_K
    rows = QUANTIZE_ROWS

    options = compile_options(spec)
    begin = time.perf_counter_ns()
    compiled = weft.compile(spec.kernel_function(), options=options,
                            toolchain=ctx.toolchain)
    compile_ms = (time.perf_counter_ns() - begin) / 1e6
    try:
        plans = argument_plans(compiled)
        plan = plans["Y"]
        record_bytes = k // plan.record_elements * plan.storage_bytes
        source = array("f", [
            (index % 31 - 15) / 16.0 for index in range(rows * k)
        ])
        outputs = [bytearray(record_bytes) for _ in range(rows)]

        arguments_by_row = []
        for row in range(rows):
            buffers = {
                "X": Buffer(source[row * k:(row + 1) * k], shape=(k,)),
                "Y": Buffer(outputs[row], shape=(k,), encoding=plan.encoding),
            }
            arguments_by_row.append(call_arguments(compiled, buffers))

        cold = Stopwatch()
        cold.measure(lambda: compiled(*arguments_by_row[0]))
        for row in range(rows):
            compiled(*arguments_by_row[row])
            expected = _REFERENCES[kind](source[row * k:(row + 1) * k])
            if bytes(outputs[row]) != expected:
                mismatch = next(
                    index for index in range(record_bytes)
                    if outputs[row][index] != expected[index])
                raise RuntimeError(
                    f"bit mismatch at byte {mismatch}: "
                    f"actual={outputs[row][mismatch]} "
                    f"expected={expected[mismatch]}")

        warm = Stopwatch()
        for _ in range(ctx.repeat):
            warm.measure(lambda: [compiled(*arguments)
                                  for arguments in arguments_by_row])
        return {
            "verdict_kind": "bitexact", "kind": kind, "k": k, "rows": rows,
            "input_policy": "fixed-float-mod31",
            "compile_ms": compile_ms, "cold_call_ms": cold.median_ms,
            "warm_median_ms": warm.median_ms, "repeat": ctx.repeat,
        }
    finally:
        compiled.close()
