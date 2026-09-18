"""Shared support for the native JIT suite: environment and target checks,
tolerance, timing, metadata-driven buffer construction and result IO."""
from __future__ import annotations

import ctypes
import json
import platform
import subprocess
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from weft.runtime import Buffer

ABSOLUTE_TOLERANCE = 1.0e-4
RELATIVE_TOLERANCE = 2.0e-3
GEMV_ABSOLUTE_TOLERANCE = 1.25e-1

QUANTIZED_PARTNER_QUANTIZER = {
    "q8_0": "q8_0", "q8_1": "q8_1", "q8_k": "q8_K",
}


class CaseFailure(Exception):
    pass


def within_tolerance(actual: float, expected: float, absolute: float) -> float:
    error = abs(actual - expected)
    if error > absolute + RELATIVE_TOLERANCE * abs(expected):
        raise CaseFailure(
            f"numeric mismatch: actual={actual!r} expected={expected!r}")
    return error


def require_finite(values, absolute_tolerance: float) -> float:
    worst = 0.0
    for actual, expected in values:
        error = within_tolerance(actual, expected, absolute_tolerance)
        worst = max(worst, error)
    return worst


def check_environment(compiler: Path, cc: Path, ggml_lib_dir: Path,
                      ggml_source: Path) -> dict:
    if platform.machine() != "riscv64":
        raise CaseFailure(f"expected riscv64 host, found {platform.machine()}")
    for path, label in (
        (compiler, "weft-compile"), (cc, "C compiler"),
        (ggml_lib_dir / "libggml-base.so", "libggml-base.so"),
        (ggml_lib_dir / "libggml-cpu.so", "libggml-cpu.so"),
        (ggml_source / "ggml/include/ggml.h", "ggml headers"),
        (ggml_source / "ggml/src/ggml-common.h", "ggml-common.h"),
    ):
        if not path.exists():
            raise CaseFailure(f"{label} is missing: {path}")
    return {
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }


def query_native_target(compiler: Path) -> dict:
    result = subprocess.run(
        [str(compiler), "--query-native-target"], capture_output=True, text=True)
    if result.returncode != 0:
        raise CaseFailure(f"--query-native-target failed: {result.stderr}")
    facts = json.loads(result.stdout)
    if facts.get("abi") != "lp64d":
        raise CaseFailure(f"unexpected ABI {facts.get('abi')!r}")
    if int(facts.get("vlen_bits", 0)) <= 0:
        raise CaseFailure("incomplete VLEN discovery")
    affinity = sorted(os_sched_getaffinity())
    if not set(affinity).issubset(facts["cpus"]):
        raise CaseFailure("current CPU affinity exceeds the discovered target")
    return facts


def os_sched_getaffinity() -> set[int]:
    import os
    return os.sched_getaffinity(0)


@dataclass(frozen=True)
class ArgumentPlan:
    name: str
    encoding: str
    shape: tuple[str, ...]
    record_elements: int
    storage_bytes: int
    alignment: int
    writable: bool


def argument_plans(compiled) -> dict[str, ArgumentPlan]:
    plans = {}
    for argument in compiled.metadata["arguments"]:
        plans[argument["name"]] = ArgumentPlan(
            name=argument["name"], encoding=argument["encoding"],
            shape=tuple(argument["shape"]),
            record_elements=argument["record_elements"],
            storage_bytes=argument["storage_bytes"],
            alignment=argument["alignment"], writable=argument["writable"],
        )
    return plans


def concrete_shape(plan: ArgumentPlan, dims: dict[str, int]) -> tuple[int, ...]:
    shape = []
    for spelling in plan.shape:
        if spelling.isdecimal():
            shape.append(int(spelling))
        elif spelling in dims:
            shape.append(dims[spelling])
        else:
            raise CaseFailure(f"unknown extent {spelling!r} for {plan.name}")
    return tuple(shape)


def encoded_storage(plan: ArgumentPlan, dims: dict[str, int]) -> int:
    shape = concrete_shape(plan, dims)
    elements = 1
    for extent in shape:
        elements *= extent
    return elements // plan.record_elements * plan.storage_bytes


def aligned_bytearray(size: int, alignment: int) -> bytearray:
    data = bytearray(size + max(alignment, 1))
    address = ctypes.addressof(ctypes.c_char.from_buffer(data))
    offset = (-address) % alignment if alignment else 0
    view = memoryview(data)[offset:offset + size]
    return bytearray(view)


def make_buffer(plan: ArgumentPlan, dims: dict[str, int], data) -> Buffer:
    shape = concrete_shape(plan, dims)
    return Buffer(data, shape=shape, encoding=plan.encoding)


def dense_buffer(values: array, dims, plan: ArgumentPlan) -> Buffer:
    return make_buffer(plan, dims, values)


class Stopwatch:
    def __init__(self) -> None:
        self.samples: list[float] = []

    def measure(self, action) -> None:
        begin = time.perf_counter_ns()
        action()
        self.samples.append((time.perf_counter_ns() - begin) / 1e6)

    @property
    def milliseconds(self) -> list[float]:
        return self.samples

    @property
    def median_ms(self) -> float:
        ordered = sorted(self.samples)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])


@dataclass
class CaseResult:
    case_id: str
    family: str
    verdict_kind: str
    status: str
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "case": self.case_id, "family": self.family,
            "verdict_kind": self.verdict_kind, "status": self.status,
            **self.detail,
        }


class ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cases = root / "cases"
        self.logs = root / "logs"
        self.cases.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

    def write_case(self, result: CaseResult) -> None:
        path = self.cases / f"{result.case_id.replace(':', '_').replace('/', '_')}.json"
        path.write_text(json.dumps(result.to_json(), sort_keys=True, indent=2),
                        encoding="utf-8")

    def write_root(self, name: str, payload) -> None:
        (self.root / name).write_text(
            json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
