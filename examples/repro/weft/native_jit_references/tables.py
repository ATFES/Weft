"""Lookup-table extraction for native JIT cases.

The iq/tq/mxfp4/nvfp4 kernels take grid, sign, codebook and scale tables as
buffer arguments.  Their values are defined by the GGML headers, so a tiny
shared object is compiled once per run against the reference GGML source tree
and the tables plus ggml_type enum values are read back through ctypes.  This
keeps the headers the single authority for table data, exactly like the
remote runner runtimes that include ggml-common.h directly.
"""
from __future__ import annotations

import ctypes
import subprocess
from array import array
from pathlib import Path

TABLE_GEN_C = r"""
#include "ggml.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <stddef.h>
#include <stdint.h>

int t_type_q1_0 = GGML_TYPE_Q1_0;
int t_type_q4_0 = GGML_TYPE_Q4_0;
int t_type_q4_1 = GGML_TYPE_Q4_1;
int t_type_q5_0 = GGML_TYPE_Q5_0;
int t_type_q5_1 = GGML_TYPE_Q5_1;
int t_type_q8_0 = GGML_TYPE_Q8_0;
int t_type_q2_k = GGML_TYPE_Q2_K;
int t_type_q3_k = GGML_TYPE_Q3_K;
int t_type_q4_k = GGML_TYPE_Q4_K;
int t_type_q5_k = GGML_TYPE_Q5_K;
int t_type_q6_k = GGML_TYPE_Q6_K;
int t_type_iq1_s = GGML_TYPE_IQ1_S;
int t_type_iq1_m = GGML_TYPE_IQ1_M;
int t_type_iq2_s = GGML_TYPE_IQ2_S;
int t_type_iq2_xs = GGML_TYPE_IQ2_XS;
int t_type_iq2_xxs = GGML_TYPE_IQ2_XXS;
int t_type_iq3_s = GGML_TYPE_IQ3_S;
int t_type_iq3_xxs = GGML_TYPE_IQ3_XXS;
int t_type_iq4_nl = GGML_TYPE_IQ4_NL;
int t_type_iq4_xs = GGML_TYPE_IQ4_XS;
int t_type_tq1_0 = GGML_TYPE_TQ1_0;
int t_type_tq2_0 = GGML_TYPE_TQ2_0;
int t_type_mxfp4 = GGML_TYPE_MXFP4;
int t_type_nvfp4 = GGML_TYPE_NVFP4;

signed char t_iq1s[2048 * 8];
signed char t_iq2s[1024 * 8];
signed char t_iq2xs[512 * 8];
signed char t_iq2xxs[256 * 8];
signed char t_iq3s[512 * 4];
signed char t_iq3xxs[256 * 4];
signed char t_signs[128 * 8];
signed char t_iq4nl[16];
signed char t_mxfp4[16];
float t_f16[65536];
float t_e8m0[256];
float t_ue4m3[256];
unsigned int t_powers[5];

static void grid64(const uint64_t *table, size_t entries, signed char *out) {
  for (size_t entry = 0; entry < entries; ++entry)
    for (size_t lane = 0; lane < 8; ++lane)
      out[entry * 8 + lane] =
          (signed char)((table[entry] >> (8 * lane)) & 0xffU);
}

static void grid32(const uint32_t *table, size_t entries, signed char *out) {
  for (size_t entry = 0; entry < entries; ++entry)
    for (size_t lane = 0; lane < 4; ++lane)
      out[entry * 4 + lane] =
          (signed char)((table[entry] >> (8 * lane)) & 0xffU);
}

__attribute__((constructor)) static void init(void) {
  grid64(iq1s_grid, 2048, t_iq1s);
  grid64(iq2s_grid, 1024, t_iq2s);
  grid64(iq2xs_grid, 512, t_iq2xs);
  grid64(iq2xxs_grid, 256, t_iq2xxs);
  grid32(iq3s_grid, 512, t_iq3s);
  grid32(iq3xxs_grid, 256, t_iq3xxs);
  for (size_t sign = 0; sign < 128; ++sign)
    for (size_t lane = 0; lane < 8; ++lane)
      t_signs[sign * 8 + lane] =
          (ksigns_iq2xs[sign] & kmask_iq2xs[lane]) ? -1 : 1;
  for (size_t index = 0; index < 16; ++index) {
    t_iq4nl[index] = kvalues_iq4nl[index];
    t_mxfp4[index] = kvalues_mxfp4[index];
  }
  for (uint32_t bits = 0; bits < 65536; ++bits) {
    const uint16_t packed = (uint16_t)bits;
    _Float16 value;
    __builtin_memcpy(&value, &packed, sizeof(value));
    t_f16[bits] = (float)value;
  }
  for (uint32_t value = 0; value < 256; ++value) {
    const uint32_t bits =
        value < 2 ? 0x00200000U << value : (value - 1U) << 23U;
    __builtin_memcpy(&t_e8m0[value], &bits, sizeof(bits));
  }
  for (uint32_t value = 0; value < 256; ++value) {
    if (value == 0 || value == 0x7fU) {
      t_ue4m3[value] = 0.0f;
      continue;
    }
    const int exponent = (int)((value >> 3U) & 0xfU);
    const int mantissa = (int)(value & 7U);
    const float raw = exponent == 0
                          ? __builtin_ldexpf((float)mantissa, -9)
                          : __builtin_ldexpf(1.0f + (float)mantissa / 8.0f,
                                             exponent - 7);
    t_ue4m3[value] = raw * 0.5f;
  }
  t_powers[0] = 1U;
  t_powers[1] = 3U;
  t_powers[2] = 9U;
  t_powers[3] = 27U;
  t_powers[4] = 81U;
}
"""

_TYPE_NAMES = (
    "q1_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
    "iq1_s", "iq1_m", "iq2_s", "iq2_xs", "iq2_xxs", "iq3_s", "iq3_xxs",
    "iq4_nl", "iq4_xs", "tq1_0", "tq2_0", "mxfp4", "nvfp4",
)

_INT8_TABLES = {
    "iq1s": 2048 * 8, "iq2s": 1024 * 8, "iq2xs": 512 * 8, "iq2xxs": 256 * 8,
    "iq3s": 512 * 4, "iq3xxs": 256 * 4, "signs": 128 * 8,
    "iq4nl": 16, "mxfp4": 16,
}


class TableLibrary:
    def __init__(self, library) -> None:
        self._library = library
        self.ggml_types = {
            name: ctypes.c_int.in_dll(library, f"t_type_{name}").value
            for name in _TYPE_NAMES
        }
        self._bytes = {}

    def raw_bytes(self, kind: str) -> bytes:
        if kind not in self._bytes:
            if kind in _INT8_TABLES:
                buffer = (ctypes.c_int8 * _INT8_TABLES[kind]).in_dll(
                    self._library, f"t_{kind}")
                self._bytes[kind] = bytes(bytearray(buffer))
            elif kind == "f16":
                buffer = (ctypes.c_float * 65536).in_dll(
                    self._library, "t_f16")
                self._bytes[kind] = array("f", buffer).tobytes()
            elif kind in ("e8m0", "ue4m3"):
                buffer = (ctypes.c_float * 256).in_dll(
                    self._library, f"t_{kind}")
                self._bytes[kind] = array("f", buffer).tobytes()
            elif kind == "powers":
                buffer = (ctypes.c_uint32 * 5).in_dll(
                    self._library, "t_powers")
                self._bytes[kind] = array("I", buffer).tobytes()
            else:
                raise KeyError(f"unknown table kind {kind}")
        return self._bytes[kind]


def build_tables(work_dir: Path, cc: str, cflags: list[str],
                 ggml_source: Path) -> TableLibrary:
    source = work_dir / "jit_tables.c"
    library_path = work_dir / "libweft_jit_tables.so"
    source.write_text(TABLE_GEN_C, encoding="utf-8")
    include_flags = [
        f"-I{ggml_source / 'ggml/include'}",
        f"-I{ggml_source / 'ggml/src'}",
    ]
    subprocess.run([
        cc, *cflags, "-O1", "-fPIC", "-shared", "-std=c11", "-Wall", "-Werror",
        *include_flags, str(source), "-o", str(library_path),
    ], check=True, capture_output=True, text=True)
    return TableLibrary(ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL))
