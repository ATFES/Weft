"""Native JIT suite runner: orchestrates every v100 binding from the tuning
catalog, audits unbound kernel files, and records per-case verdicts.

Lifecycle per executable case:
    weft.compile (timed) -> build declared inputs -> cold call (timed)
    -> numeric/bitexact verification -> warm repeats (timed)
Suite-level checks cover same-binding reuse, different-binding specialization
and the Buffer ABI contract.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
for entry in (PROJECT / "python", PROJECT / "examples"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import weft  # noqa: E402
from weft.diagnostics import WeftError  # noqa: E402
from weft.runtime import Buffer, CompileOptions, Toolchain  # noqa: E402

from native_jit_common import (  # noqa: E402
    CaseFailure, CaseResult, ResultStore, check_environment,
    query_native_target, require_finite, Stopwatch, ABSOLUTE_TOLERANCE,
)
from native_jit_references import GgmlReference, build_tables  # noqa: E402
from native_jit_registry import load_registry, select  # noqa: E402
from native_jit_cases import dense, dequantize, gemv, mul_mat, quantize, vec_dot  # noqa: E402

FAMILY_EXECUTORS = {
    "dense": dense.execute,
    "vec_dot": vec_dot.execute,
    "dequantize": dequantize.execute,
    "gemv": gemv.execute,
    "mul_mat": mul_mat.execute,
    "quantize": quantize.execute,
}


@dataclass
class SuiteContext:
    toolchain: Toolchain
    ggml: GgmlReference
    tables: object
    repeat: int
    store: ResultStore


def run_binding_case(spec, ctx: SuiteContext) -> CaseResult:
    try:
        executor = FAMILY_EXECUTORS[spec.family]
        detail = executor(spec, ctx)
        detail["runner"] = spec.runner
        detail["entry"] = spec.entry.replace("kernels.", "").replace(".", "/")
        detail["symbol"] = spec.symbol
        detail["binding_origin"] = f"tuning:{spec.runner}:v100 {spec.request}"
        return CaseResult(spec.case_id, spec.family,
                          detail.get("verdict_kind", "numeric"), "PASS",
                          detail)
    except CaseFailure as failure:
        return _failure(spec, failure)
    except WeftError as failure:
        return _failure(spec, failure)
    except Exception as failure:  # noqa: BLE001
        return _failure(spec, failure, traceback.format_exc())


def _failure(spec, failure, trace: str | None = None) -> CaseResult:
    log = f"{failure.__class__.__name__}: {failure}\n"
    if trace:
        log += trace
    return CaseResult(spec.case_id, spec.family, "error", "FAIL", {
        "error": log.strip().splitlines()[-1] if log.strip() else "failure",
        "runner": spec.runner,
        "detail_log": log,
    })


def run_no_binding_case(spec, ctx: SuiteContext) -> CaseResult:
    detail = {
        "entry": spec.entry.replace("kernels.", "").replace(".", "/"),
        "declared_targets": list(spec.declared_targets),
    }
    if spec.entry == "kernels.dense.ime_i8_contract":
        detail["native_contract"] = _ime_contract_check(ctx)
        if not detail["native_contract"]["raised"]:
            return CaseResult(spec.case_id, spec.family,
                              "no-v100-binding", "FAIL", detail)
    return CaseResult(spec.case_id, spec.family, "no-v100-binding", "PASS",
                      detail)


def _ime_contract_check(ctx: SuiteContext) -> dict:
    catalog = json.loads(
        (PROJECT / "examples/kernels/dense/tuning.json").read_text())
    binding = catalog["bindings"]["kernel:k1 ime_i8_contract"]
    from kernels.dense.ime_i8_contract import ime_i8_contract
    try:
        compiled = weft.compile(
            ime_i8_contract,
            options=CompileOptions(meta=binding["source"]),
            toolchain=ctx.toolchain)
    except (WeftError, NotImplementedError) as failure:
        compiled = None
        return {"raised": True,
                "error": f"{failure.__class__.__name__}: {failure}"}
    compiled.close()
    return {"raised": False,
            "error": "matrix kernel compiled without declared v100 support"}


def suite_checks(ctx: SuiteContext) -> dict:
    from kernels.dense.gemm_f32 import gemm_f32

    checks: dict = {}
    m, n, k = 16, 32, 256
    options = CompileOptions(meta={"MC": 64, "NC": 16, "MR": 2, "NR": 2},
                             lmul_eighths=16)
    dense = weft.compile(gemm_f32, options=options, toolchain=ctx.toolchain)
    try:
        lhs = __import__("array").array("f", [1.0 / k]) * (m * k)
        rhs = __import__("array").array("f", [1.0]) * (n * k)
        output = __import__("array").array("f", [0.0]) * (m * n)
        arguments = (Buffer(lhs, shape=(m, k)), Buffer(rhs, shape=(n, k)),
                     Buffer(output, shape=(m, n)))
        dense(*arguments)
        checks["dense_numeric"] = require_finite(
            ((value, 1.0) for value in output), ABSOLUTE_TOLERANCE)

        again = weft.compile(gemm_f32, options=options,
                             toolchain=ctx.toolchain)
        checks["same_binding_reused"] = again is dense
        alternative = weft.compile(
            gemm_f32,
            options=CompileOptions(meta=options.meta, lmul_eighths=16,
                                   unroll=2),
            toolchain=ctx.toolchain)
        checks["different_binding_specialized"] = alternative is not dense
        try:
            alternative(*arguments)
            checks["specialized_numeric"] = require_finite(
                ((value, 1.0) for value in output), ABSOLUTE_TOLERANCE)
        finally:
            alternative.close()

        checks["abi_contract"] = _abi_checks(dense, m, n, k)
    finally:
        dense.close()

    quantizer = None
    try:
        checks["abi_contract"]["alias_overlap"] = _alias_check(ctx)
    finally:
        if quantizer is not None:
            quantizer.close()
    return checks


def _expect(exception_type, action) -> dict:
    try:
        action()
    except exception_type as failure:
        return {"raised": True, "error": f"{type(failure).__name__}"}
    except Exception as failure:  # noqa: BLE001
        return {"raised": False,
                "error": f"unexpected {type(failure).__name__}: {failure}"}
    return {"raised": False, "error": "no error raised"}


def _abi_checks(dense, m: int, n: int, k: int) -> dict:
    from array import array
    lhs = array("f", [1.0 / k]) * (m * k)
    rhs = array("f", [1.0]) * (n * k)
    good_output = array("f", bytes(4 * m * n))
    checks = {
        "wrong_rank": _expect(ValueError, lambda: dense(
            Buffer(lhs, shape=(m, k)), Buffer(rhs, shape=(n, k)),
            Buffer(good_output, shape=(m * n,)))),
        "wrong_encoding": _expect(TypeError, lambda: dense(
            Buffer(lhs, shape=(m, k)), Buffer(rhs, shape=(n, k)),
            Buffer(bytearray(34 * m * n), shape=(m, n),
                   encoding="Q8_0"))),
        "readonly_output": _expect(TypeError, lambda: dense(
            Buffer(lhs, shape=(m, k)), Buffer(rhs, shape=(n, k)),
            Buffer(bytes(4 * m * n), shape=(m, n)))),
        "short_storage": _expect(ValueError, lambda: dense(
            Buffer(lhs, shape=(m, k)),
            Buffer(array("f", [0.0]) * (n * k - 1), shape=(n, k)),
            Buffer(good_output, shape=(m, n)))),
        "misaligned": _expect(ValueError, lambda: dense(
            Buffer(memoryview(bytearray(4 * m * k + 1))[1:], shape=(m, k),
                   encoding="dense.f32"),
            Buffer(rhs, shape=(n, k)), Buffer(good_output, shape=(m, n)))),
    }
    return checks


def _alias_check(ctx) -> dict:
    from kernels.mul_mat.q4_k_decode import production_mul_mat_q4_k_decode
    kernel = weft.compile(production_mul_mat_q4_k_decode,
                          toolchain=ctx.toolchain)
    try:
        n, k = 2, 256
        weights = bytearray(144 * n)
        shared = bytearray(4 * k + 292)
        arguments = (
            Buffer(weights, shape=(n, k), encoding="Q4_K"),
            Buffer(memoryview(shared), shape=(1, k), encoding="dense.f32"),
            Buffer(memoryview(shared), shape=(1, k), encoding="Q8_K"),
            Buffer(bytearray(4 * n), shape=(1, n)),
        )
        return _expect(ValueError, lambda: kernel(*arguments))
    finally:
        kernel.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True,
                        help="case id, family prefix, or all")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cflag", action="append", default=[])
    parser.add_argument("--ggml-lib-dir", type=Path,
                        default=Path("/home/zhy001/rvv-v100-llama"
                                     "/build-v100-clang18-gomp/bin"))
    parser.add_argument("--ggml-source-dir", type=Path,
                        default=Path("/home/zhy001/rvv-v100-llama"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("/tmp/weft-native-jit"))
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    store = ResultStore(args.result_dir / run_id)
    environment = check_environment(Path(args.compiler), Path(args.cc),
                                    args.ggml_lib_dir, args.ggml_source_dir)
    target = query_native_target(Path(args.compiler))
    store.write_root("environment.json", environment)
    store.write_root("target.json", target)

    work = store.root / "work"
    work.mkdir(parents=True, exist_ok=True)
    tables = build_tables(work, args.cc, args.cflag, args.ggml_source_dir)
    ggml = GgmlReference(args.ggml_lib_dir)
    toolchain = Toolchain(args.compiler, (args.cc,), tuple(args.cflag))
    ctx = SuiteContext(toolchain, ggml, tables, args.repeat, store)

    bindings, unbound = load_registry(PROJECT)
    selected_bindings, selected_unbound = select(args.case, bindings, unbound)
    store.write_root("audit.json", {
        "total_v100_bindings": len(bindings),
        "total_kernel_files": len(list(
            PROJECT.glob("examples/kernels/*/*.py"))),
        "unbound_files": len(unbound),
        "selected_bindings": len(selected_bindings),
        "selected_unbound": len(selected_unbound),
    })

    results = []
    for spec in selected_bindings:
        result = run_binding_case(spec, ctx)
        results.append(result)
        store.write_case(result)
        if result.status == "FAIL":
            log = result.detail.get("detail_log", "")
            (store.logs / f"{result.case_id.replace(':', '_')}.log").write_text(
                log, encoding="utf-8")
        print(f"{result.status:4} {result.case_id}", flush=True)
    for spec in selected_unbound:
        result = run_no_binding_case(spec, ctx)
        results.append(result)
        store.write_case(result)
        print(f"{result.status:4} {result.case_id}", flush=True)

    suite = {}
    if args.case == "all":
        suite = suite_checks(ctx)
        store.write_root("suite.json", suite)
        print(f"suite checks: {json.dumps(suite, sort_keys=True)[:400]}",
              flush=True)

    summary = {
        "pass": sum(1 for r in results if r.status == "PASS"),
        "fail": sum(1 for r in results if r.status == "FAIL"),
        "by_verdict_kind": {
            kind: sum(1 for r in results if r.verdict_kind == kind)
            for kind in sorted({r.verdict_kind for r in results})
        },
    }
    payload = {
        "schema_version": 1,
        "suite": "native-jit",
        "run_id": run_id,
        "target": {"march": target["march"], "abi": target["abi"],
                   "vlen_bits": target["vlen_bits"]},
        "repeat": args.repeat,
        "summary": summary,
        "suite_checks": suite,
        "cases": [r.to_json() for r in results],
    }
    store.write_root("result.json", payload)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["case", "family", "verdict_kind", "status",
                     "max_absolute_error", "compile_ms", "cold_call_ms",
                     "warm_median_ms"])
    for result in results:
        detail = result.detail
        writer.writerow([
            result.case_id, result.family, result.verdict_kind, result.status,
            detail.get("max_absolute_error", ""),
            detail.get("compile_ms", ""), detail.get("cold_call_ms", ""),
            detail.get("warm_median_ms", ""),
        ])
    (store.root / "summary.csv").write_text(buffer.getvalue(), encoding="utf-8")

    print(f"results: {store.root}", flush=True)
    print(f"summary: {json.dumps(summary, sort_keys=True)}", flush=True)
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
