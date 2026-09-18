"""ctypes bindings for the GGML reference libraries used by the native JIT suite.

GGML is the reference authority for quantized formats: quantized inputs are
produced with ggml_quantize_chunk / quantize_row_q8_*_ref and expected outputs
are computed with ggml_vec_dot_* / dequantize_row_*, mirroring the remote
runner runtimes in examples/repro/weft/*_runtime.cpp.
"""
from __future__ import annotations

import ctypes
from array import array
from pathlib import Path


_INT64 = ctypes.c_int64
_SIZE_T = ctypes.c_size_t


def _address(data) -> int:
    if isinstance(data, array):
        return data.buffer_info()[0]
    return ctypes.addressof((ctypes.c_char * len(data)).from_buffer(data))


class GgmlReference:
    def __init__(self, lib_dir: Path) -> None:
        self.base = ctypes.CDLL(str(lib_dir / "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
        self.cpu = ctypes.CDLL(str(lib_dir / "libggml-cpu.so"), mode=ctypes.RTLD_GLOBAL)

        self.quantize_chunk = self.base.ggml_quantize_chunk
        self.quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, _INT64, _INT64,
            _INT64, ctypes.c_void_p,
        ]
        self.quantize_chunk.restype = _INT64

        for name in ("quantize_row_q8_0_ref", "quantize_row_q8_1_ref", "quantize_row_q8_K_ref"):
            function = getattr(self.base, name)
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _INT64]
            function.restype = None

        self._dequantize = {}
        self._vec_dot = {}
        self._importance: dict[int, array] = {}

    def quantize_record(self, ggml_type: int, source, record) -> int:
        """Quantize one record of len(source) elements; returns encoded byte count.

        The importance vector is all ones, matching the remote runner policy;
        iq-format quantizers require a non-NULL imatrix.
        """
        count = len(source)
        if count not in self._importance:
            self._importance[count] = array("f", [1.0]) * count
        return int(self.quantize_chunk(
            ggml_type, _address(source), _address(record),
            0, 1, count, _address(self._importance[count]),
        ))

    _PARTNER_SYMBOL = {"q8_0": "q8_0", "q8_1": "q8_1", "q8_k": "q8_K"}

    def quantize_row(self, partner: str, source, destination, count: int) -> None:
        function = getattr(self.base,
                           f"quantize_row_{self._PARTNER_SYMBOL[partner]}_ref")
        function(_address(source), _address(destination), count)

    def dequantize(self, format_key: str) -> ctypes._NamedFuncPointer:
        symbol = f"dequantize_row_{format_key}"
        if symbol not in self._dequantize:
            function = getattr(self.base, symbol)
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _INT64]
            function.restype = None
            self._dequantize[symbol] = function
        return self._dequantize[symbol]

    def vec_dot(self, symbol: str) -> ctypes._NamedFuncPointer:
        if symbol not in self._vec_dot:
            function = getattr(self.cpu, symbol)
            function.argtypes = [
                ctypes.c_int, ctypes.c_void_p, _SIZE_T,
                ctypes.c_void_p, _SIZE_T, ctypes.c_void_p, _SIZE_T, ctypes.c_int,
            ]
            function.restype = None
            self._vec_dot[symbol] = function
        return self._vec_dot[symbol]

    def reference_dot(self, symbol: str, n: int, weights, activation) -> float:
        result = ctypes.c_float()
        self.vec_dot(symbol)(
            n, ctypes.byref(result), 0,
            _address(weights), 0, _address(activation), 0, 1,
        )
        return result.value

    def reference_dequantize(self, format_key: str, record, output, count: int) -> None:
        self.dequantize(format_key)(_address(record), _address(output), count)
