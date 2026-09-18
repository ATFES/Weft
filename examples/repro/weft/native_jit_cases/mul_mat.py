"""mul_mat family: production matrix-multiply cases for every v100 binding.

Mirrors mul_mat_runtime.cpp: N=K=4096, decode M=1, prefill M=128.  Weights are
one ggml-quantized record replicated over K and N; activations use the fixed
(37i mod 251) pattern; the reference is the ggml vec-dot of the weight record
row against the reference-quantized activation row.  The persistent variant
packs weights through the kernel-exported W_pack entrypoint.  The f16 variant
additionally validates its f16 staging workspace bit-exactly.
"""
from __future__ import annotations

import ctypes
import struct
import time
from array import array

import weft
from weft.runtime import Buffer

from native_jit_common import (
    ABSOLUTE_TOLERANCE, Stopwatch, argument_plans, require_finite,
)
from native_jit_cases import (
    FORMAT_INFO, FORMAT_ORDER, PARTNER_RECORD_BYTES, TABLE_PARAM_NAMES,
    activation_source, call_arguments, compile_options, table_buffer,
    weight_source,
)

MUL_MAT_N = 4096
MUL_MAT_K = 4096


def _format_of(request: str) -> str:
    base = request.rsplit(" ", 1)[0]
    if base == "f16":
        return "f16"
    for suffix in ("_staged", "_persistent"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if base not in FORMAT_INFO:
        raise KeyError(f"unknown mul-mat format base {base!r}")
    return base


def _pack_weights(compiled, symbol: str, weights: array, rows: int,
                  cols: int) -> bytearray:
    library = ctypes.CDLL(str(compiled.library_path))
    size_fn = library[f"{symbol}_W_packed_size"]
    size_fn.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    size_fn.restype = ctypes.c_size_t
    pack_fn = library[f"{symbol}_W_pack"]
    pack_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_size_t, ctypes.c_size_t]
    pack_fn.restype = None
    packed = bytearray(size_fn(rows, cols))
    pack_fn(weights.buffer_info()[0],
            ctypes.addressof((ctypes.c_char * len(packed)).from_buffer(packed)),
            rows, cols)
    return packed


def execute(spec, ctx) -> dict:
    fmt = _format_of(spec.request)
    decode = spec.request.endswith("decode")
    m = 1 if decode else 128
    n, k = MUL_MAT_N, MUL_MAT_K

    options = compile_options(spec)
    begin = time.perf_counter_ns()
    compiled = weft.compile(spec.kernel_function(), options=options,
                            toolchain=ctx.toolchain)
    compile_ms = (time.perf_counter_ns() - begin) / 1e6
    try:
        plans = argument_plans(compiled)
        buffers = {}

        if fmt == "f16":
            weight_values = [
                ((index * 19) % 127 - 63) / 13.0 for index in range(k)
            ]
            weight_row = b"".join(
                struct.pack("<e", value) for value in weight_values)
            weights = bytearray(weight_row * n)
            activation = activation_source(m * k)
            staging = bytearray(2 * m * k)
            output = array("f", bytes(4 * m * n))
            buffers["W"] = Buffer(weights, shape=(n, k),
                                  encoding=plans["W"].encoding)
            buffers["X"] = Buffer(activation, shape=(m, k))
            buffers["Xh"] = Buffer(staging, shape=(m, k),
                                   encoding=plans["Xh"].encoding)
            buffers["Y"] = Buffer(output, shape=(m, n))
            weight_row_values = array("f", [
                struct.unpack("<e", weight_row[index * 2:index * 2 + 2])[0]
                for index in range(k)
            ])
            activation16 = [
                struct.unpack("<e", struct.pack("<e", value))[0]
                for value in activation
            ]
            expected_rows = []
            for row in range(m):
                total = 0.0
                for index in range(k):
                    total += weight_row_values[index] * \
                        activation16[row * k + index]
                expected_rows.append(total)
            expected_staging = b"".join(
                struct.pack("<e", value) for value in activation)
            workspace_check = "bit-exact"
        else:
            info = FORMAT_INFO[fmt]
            format_index = FORMAT_ORDER.index(fmt) + 1
            source = weight_source(format_index, info.wqk)
            scratch = bytearray(info.wqk * 4)
            record_bytes = ctx.ggml.quantize_record(
                ctx.tables.ggml_types[fmt], source, scratch)
            plan = plans["W"]
            expected_record = info.wqk // plan.record_elements * \
                plan.storage_bytes
            if record_bytes != expected_record:
                raise AssertionError(
                    f"quantized record is {record_bytes} bytes, kernel "
                    f"expects {expected_record}")
            weight_row = array("B", scratch[:record_bytes]) * (k // info.wqk)
            weights = array("B", weight_row.tobytes() * n)

            activation = activation_source(m * k)
            partner_record = PARTNER_RECORD_BYTES[info.partner]
            row_records = k // info.xqk
            reference_workspace = array("B", bytes(partner_record * row_records))
            expected_rows = []
            for row in range(m):
                ctx.ggml.quantize_row(
                    info.partner,
                    activation[row * k:(row + 1) * k],
                    reference_workspace, k)
                expected_rows.append(ctx.ggml.reference_dot(
                    info.vec_dot, k, weight_row, reference_workspace))

            buffers["W"] = Buffer(weights, shape=(n, k),
                                  encoding=info.weight_encoding)
            buffers["X"] = Buffer(activation, shape=(m, k))
            buffers["Xq"] = Buffer(
                array("B", bytes(partner_record * row_records * m)),
                shape=(m, k), encoding=info.partner_encoding)
            output = array("f", bytes(4 * m * n))
            buffers["Y"] = Buffer(output, shape=(m, n))
            workspace_check = "numeric-only"

        for name in TABLE_PARAM_NAMES:
            if name in plans:
                buffers[name] = table_buffer(plans[name], ctx.tables, fmt)

        if spec.entry.endswith("_persistent"):
            packed = _pack_weights(compiled, spec.symbol, weights, n, k)
            buffers["W"] = Buffer(packed, shape=(n, k),
                                  encoding=plans["W"].encoding)

        arguments = call_arguments(compiled, buffers)
        cold = Stopwatch()
        cold.measure(lambda: compiled(*arguments))
        if fmt == "f16" and bytes(staging) != expected_staging:
            raise RuntimeError("f16 staging workspace mismatch")
        expected_values = [expected_rows[row // n] for row in range(m * n)]
        max_error = require_finite(zip(output, expected_values),
                                   ABSOLUTE_TOLERANCE)

        warm = Stopwatch()
        for _ in range(ctx.repeat):
            warm.measure(lambda: compiled(*arguments))
        return {
            "verdict_kind": "numeric", "format": fmt,
            "phase": "decode" if decode else "prefill",
            "m": m, "n": n, "k": k,
            "input_policy": "valid-quantized-record-replicated",
            "workspace_check": workspace_check,
            "max_absolute_error": max_error, "compile_ms": compile_ms,
            "cold_call_ms": cold.median_ms, "warm_median_ms": warm.median_ms,
            "repeat": ctx.repeat,
        }
    finally:
        compiled.close()
