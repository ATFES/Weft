"""gemv family: quantized q4_k gemv variants on interleaved packed weights.

Mirrors q4_k_gemv_runtime.cpp: M=14336, K=4096, pseudo-random Q4_K weight
records and Q8_K activation records with bounded scales, weights packed via
the kernel-exported W_pack entrypoint, reference computed per row with
ggml_vec_dot_q4_K_q8_K, and the gemv-specific absolute tolerance.
"""
from __future__ import annotations

import ctypes
import random
import struct
import time
from array import array

import weft
from weft.runtime import Buffer

from native_jit_common import (
    GEMV_ABSOLUTE_TOLERANCE, Stopwatch, argument_plans, require_finite,
)
from native_jit_cases import call_arguments, compile_options

GEMV_M = 14336
GEMV_K = 4096
Q4_K_RECORD_BYTES = 144
Q8_K_RECORD_BYTES = 292
BLOCK = 256
VEC_DOT = "ggml_vec_dot_q4_K_q8_K_generic"


def execute(spec, ctx) -> dict:
    m, k = GEMV_M, GEMV_K
    blocks = k // BLOCK

    options = compile_options(spec)
    begin = time.perf_counter_ns()
    compiled = weft.compile(spec.kernel_function(), options=options,
                            toolchain=ctx.toolchain)
    compile_ms = (time.perf_counter_ns() - begin) / 1e6
    try:
        generator = random.Random(0x4B1D2A73)
        weights = array("B", generator.randbytes(m * blocks * Q4_K_RECORD_BYTES))
        for record in range(m * blocks):
            base = record * Q4_K_RECORD_BYTES
            d = 0.125 + generator.getrandbits(5) / 128.0
            dmin = 0.0625 + generator.getrandbits(4) / 256.0
            struct.pack_into("<e", weights, base, d)
            struct.pack_into("<e", weights, base + 2, dmin)

        activation_generator = random.Random(0x8AC6F251)
        activation = array("B", activation_generator.randbytes(
            blocks * Q8_K_RECORD_BYTES))
        for block in range(blocks):
            base = block * Q8_K_RECORD_BYTES
            scale = 0.125 + activation_generator.getrandbits(5) / 128.0
            struct.pack_into("<f", activation, base, scale)

        expected = []
        row_bytes = blocks * Q4_K_RECORD_BYTES
        for row in range(m):
            expected.append(ctx.ggml.reference_dot(
                VEC_DOT, k,
                weights[row * row_bytes:(row + 1) * row_bytes], activation))

        library = ctypes.CDLL(str(compiled.library_path))
        size_fn = library[f"{spec.symbol}_W_packed_size"]
        size_fn.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        size_fn.restype = ctypes.c_size_t
        pack_fn = library[f"{spec.symbol}_W_pack"]
        pack_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_size_t, ctypes.c_size_t]
        pack_fn.restype = None
        packed = bytearray(size_fn(m, k))
        pack_fn(weights.buffer_info()[0],
                ctypes.addressof(
                    (ctypes.c_char * len(packed)).from_buffer(packed)),
                m, k)

        plans = argument_plans(compiled)
        output = array("f", bytes(4 * m))
        buffers = {
            "W": Buffer(packed, shape=(m, k), encoding=plans["W"].encoding),
            "X": Buffer(activation, shape=(k,), encoding="Q8_K"),
            "Y": Buffer(output, shape=(m,)),
        }
        arguments = call_arguments(compiled, buffers)

        cold = Stopwatch()
        cold.measure(lambda: compiled(*arguments))
        max_error = require_finite(zip(output, expected),
                                   GEMV_ABSOLUTE_TOLERANCE)

        warm = Stopwatch()
        for _ in range(ctx.repeat):
            warm.measure(lambda: compiled(*arguments))
        return {
            "verdict_kind": "numeric", "format": "q4_k", "m": m, "k": k,
            "input_policy": "random-valid-records-packed",
            "max_absolute_error": max_error, "compile_ms": compile_ms,
            "cold_call_ms": cold.median_ms, "warm_median_ms": warm.median_ms,
            "repeat": ctx.repeat,
        }
    finally:
        compiled.close()
