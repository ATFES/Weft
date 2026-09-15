"""Family case executors and the shared quantized-format table.

Format facts mirror the remote runner runtimes in examples/repro/weft:
FORMAT_ORDER matches kWeightTypes ordering in vec_dot/mul_mat/row_dequantize
runtimes, partners match the runner format tables, wqk values match the
selected_wqk macros and vec-dot/dequantize symbols match the selected
reference macros.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass

from weft.runtime import Buffer


FORMAT_ORDER = (
    "q1_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
    "iq1_s", "iq1_m", "iq2_s", "iq2_xs", "iq2_xxs", "iq3_s", "iq3_xxs",
    "iq4_nl", "iq4_xs", "tq1_0", "tq2_0", "mxfp4", "nvfp4",
)


@dataclass(frozen=True)
class FormatInfo:
    weight_encoding: str
    wqk: int
    partner: str          # q8_0 | q8_1 | q8_k
    partner_encoding: str
    xqk: int
    vec_dot: str
    dequantize: str


def _info(weight_encoding, wqk, partner, partner_encoding, xqk, vec_dot, dequantize):
    return FormatInfo(weight_encoding, wqk, partner, partner_encoding, xqk,
                      vec_dot, dequantize)


FORMAT_INFO = {
    "q1_0": _info("Q1_0", 128, "q8_0", "Q8_0", 32,
                  "ggml_vec_dot_q1_0_q8_0_generic", "q1_0"),
    "q4_0": _info("Q4_0", 32, "q8_0", "Q8_0", 32,
                  "ggml_vec_dot_q4_0_q8_0_generic", "q4_0"),
    "q4_1": _info("Q4_1", 32, "q8_1", "Q8_1", 32,
                  "ggml_vec_dot_q4_1_q8_1_generic", "q4_1"),
    "q5_0": _info("Q5_0", 32, "q8_0", "Q8_0", 32,
                  "ggml_vec_dot_q5_0_q8_0_generic", "q5_0"),
    "q5_1": _info("Q5_1", 32, "q8_1", "Q8_1", 32,
                  "ggml_vec_dot_q5_1_q8_1_generic", "q5_1"),
    "q8_0": _info("Q8_0", 32, "q8_0", "Q8_0", 32,
                  "ggml_vec_dot_q8_0_q8_0_generic", "q8_0"),
    "q2_k": _info("Q2_K", 256, "q8_k", "Q8_K", 256,
                  "ggml_vec_dot_q2_K_q8_K_generic", "q2_K"),
    "q3_k": _info("Q3_K", 256, "q8_k", "Q8_K", 256,
                  "ggml_vec_dot_q3_K_q8_K_generic", "q3_K"),
    "q4_k": _info("Q4_K", 256, "q8_k", "Q8_K", 256,
                  "ggml_vec_dot_q4_K_q8_K_generic", "q4_K"),
    "q5_k": _info("Q5_K", 256, "q8_k", "Q8_K", 256,
                  "ggml_vec_dot_q5_K_q8_K_generic", "q5_K"),
    "q6_k": _info("Q6_K", 256, "q8_k", "Q8_K", 256,
                  "ggml_vec_dot_q6_K_q8_K_generic", "q6_K"),
    "iq1_s": _info("IQ1_S", 256, "q8_k", "Q8_K", 256,
                   "ggml_vec_dot_iq1_s_q8_K_generic", "iq1_s"),
    "iq1_m": _info("IQ1_M", 256, "q8_k", "Q8_K", 256,
                   "ggml_vec_dot_iq1_m_q8_K_generic", "iq1_m"),
    "iq2_s": _info("IQ2_S", 256, "q8_k", "Q8_K", 256,
                   "ggml_vec_dot_iq2_s_q8_K_generic", "iq2_s"),
    "iq2_xs": _info("IQ2_XS", 256, "q8_k", "Q8_K", 256,
                    "ggml_vec_dot_iq2_xs_q8_K_generic", "iq2_xs"),
    "iq2_xxs": _info("IQ2_XXS", 256, "q8_k", "Q8_K", 256,
                     "ggml_vec_dot_iq2_xxs_q8_K_generic", "iq2_xxs"),
    "iq3_s": _info("IQ3_S", 256, "q8_k", "Q8_K", 256,
                   "ggml_vec_dot_iq3_s_q8_K_generic", "iq3_s"),
    "iq3_xxs": _info("IQ3_XXS", 256, "q8_k", "Q8_K", 256,
                     "ggml_vec_dot_iq3_xxs_q8_K_generic", "iq3_xxs"),
    "iq4_nl": _info("IQ4_NL", 32, "q8_0", "Q8_0", 32,
                    "ggml_vec_dot_iq4_nl_q8_0_generic", "iq4_nl"),
    "iq4_xs": _info("IQ4_XS", 256, "q8_k", "Q8_K", 256,
                    "ggml_vec_dot_iq4_xs_q8_K_generic", "iq4_xs"),
    "tq1_0": _info("TQ1_0", 256, "q8_k", "Q8_K", 256,
                   "ggml_vec_dot_tq1_0_q8_K_generic", "tq1_0"),
    "tq2_0": _info("TQ2_0", 256, "q8_k", "Q8_K", 256,
                   "ggml_vec_dot_tq2_0_q8_K_generic", "tq2_0"),
    "mxfp4": _info("MXFP4", 32, "q8_0", "Q8_0", 32,
                   "ggml_vec_dot_mxfp4_q8_0_generic", "mxfp4"),
    "nvfp4": _info("NVFP4", 64, "q8_0", "Q8_0", 32,
                   "ggml_vec_dot_nvfp4_q8_0", "nvfp4"),
}

TABLE_PARAM_NAMES = frozenset(
    {"grid", "signs", "codebook", "scale", "f16_bits", "powers"})

_GRID_KIND = {
    "iq1_s": "iq1s", "iq1_m": "iq1s", "iq2_s": "iq2s", "iq2_xs": "iq2xs",
    "iq2_xxs": "iq2xxs", "iq3_s": "iq3s", "iq3_xxs": "iq3xxs",
}


def table_kind(fmt: str, name: str) -> str:
    if name == "grid":
        return _GRID_KIND[fmt]
    if name == "signs":
        return "signs"
    if name == "codebook":
        return "iq4nl" if fmt.startswith("iq4") else "mxfp4"
    if name == "scale":
        return "ue4m3" if fmt == "nvfp4" else "e8m0"
    if name == "f16_bits":
        return "f16"
    if name == "powers":
        return "powers"
    raise KeyError(name)


def table_buffer(plan, tables, fmt: str):
    """Build the table buffer for one lookup-table argument plan."""
    raw = bytearray(tables.raw_bytes(table_kind(fmt, plan.name)))
    if plan.encoding in ("I8X8", "I8X4"):
        data = raw
    elif plan.encoding == "dense.i8":
        data = array("b", raw)
    elif plan.encoding == "dense.f32":
        data = array("f", raw)
    elif plan.encoding == "dense.u32":
        data = array("I", raw)
    else:
        raise ValueError(f"unexpected table encoding {plan.encoding}")
    shape = []
    for spelling in plan.shape:
        if not spelling.isdecimal():
            raise ValueError(f"table extent must be concrete: {plan.name}")
        shape.append(int(spelling))
    return Buffer(data, shape=tuple(shape), encoding=plan.encoding)


def weight_source(format_index: int, wqk: int) -> array:
    return array("f", [
        ((index * 19 + format_index * 7) % 127 - 63) / 13.0
        for index in range(wqk)
    ])


def activation_source(count: int, offset: int = 0) -> array:
    return array("f", [
        (((index + offset) * 37) % 251 - 125) / 17.0
        for index in range(count)
    ])


def compile_options(spec):
    from weft.runtime import CompileOptions
    return CompileOptions(meta=spec.meta, **spec.physical)


def call_arguments(compiled, buffers_by_name) -> list:
    names = [argument["name"] for argument in compiled.metadata["arguments"]]
    return [buffers_by_name[name] for name in names]


PARTNER_RECORD_BYTES = {"q8_0": 34, "q8_1": 36, "q8_k": 292}
