"""Registry for the native JIT suite.

The authoritative test surface is the set of `*:v100` bindings declared in
examples/kernels/*/tuning.json.  Every kernel file under examples/kernels must
be claimed by at least one binding or it is reported as no-v100-binding; the
audit is bidirectional so nothing is silently skipped.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

PHYSICAL_FIELDS = (
    "lmul_eighths", "unroll", "pipeline_depth", "scalar_load_prime",
    "partial_combine_policy", "record_axis_policy",
)

FAMILY_RUNNERS = {
    "dense": ("kernel", "mul-mat"),
    "dequantize": ("row-dequantize",),
    "gemv": ("kernel",),
    "mul_mat": ("mul-mat",),
    "quantize": ("kernel",),
    "vec_dot": ("vec-dot",),
}


@dataclass(frozen=True)
class BindingSpec:
    case_id: str
    family: str
    runner: str
    request: str
    entry: str          # module path, e.g. kernels.dense.gemm_f32
    symbol: str
    meta: dict
    physical: dict

    def kernel_function(self):
        module = importlib.import_module(self.entry)
        return getattr(module, self.symbol)


@dataclass(frozen=True)
class NoBindingSpec:
    case_id: str
    family: str
    entry: str          # module path of the kernel file
    declared_targets: tuple[str, ...]


def _module_path(project: Path, entry_file: Path) -> str:
    parts = entry_file.relative_to(project).with_suffix("").parts
    if parts[0] == "examples":
        parts = parts[1:]
    return ".".join(parts)


def load_registry(project: Path, target: str = "v100"):
    project = Path(project)
    bindings = []
    claimed = set()
    declared_by_file: dict[str, set[str]] = {}
    for catalog_path in sorted(project.glob("examples/kernels/*/tuning.json")):
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        family = catalog_path.parent.name
        for key, binding in catalog["bindings"].items():
            runner, _, rest = key.partition(":")
            target_of, _, request = rest.partition(" ")
            entry_file = catalog_path.parent / binding["entry"]
            module = _module_path(project, entry_file)
            declared_by_file.setdefault(module, set()).add(runner)
            if target_of != target or not request:
                continue
            physical = dict(catalog["physical_defaults"], **binding["physical"])
            if set(physical) != set(PHYSICAL_FIELDS):
                raise ValueError(f"incomplete physical binding for {key}")
            request_id = request.replace(" ", "-")
            bindings.append(BindingSpec(
                case_id=f"{family}:{request_id}", family=family,
                runner=runner, request=request, entry=module,
                symbol=binding["symbol"], meta=dict(binding["source"]),
                physical=physical,
            ))
            claimed.add(module)
            declared_by_file.setdefault(module, set()).discard(runner)

    unbound = []
    for entry_file in sorted(project.glob("examples/kernels/*/*.py")):
        module = _module_path(project, entry_file)
        if module in claimed:
            continue
        family = entry_file.parent.name
        targets = sorted({
            key.partition(":")[2].partition(" ")[0]
            for key in _file_targets(project, family, entry_file.name)
            if key.partition(":")[2].partition(" ")[0] != target
        })
        unbound.append(NoBindingSpec(
            case_id=f"nobinding:{family}.{entry_file.stem}",
            family=family, entry=module,
            declared_targets=tuple(t for t in targets if t != target),
        ))
    return bindings, unbound


def _file_targets(project: Path, family: str, filename: str) -> list[str]:
    catalog_path = project / "examples/kernels" / family / "tuning.json"
    if not catalog_path.exists():
        return []
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return [key for key, binding in catalog["bindings"].items()
            if binding["entry"] == filename]


def select(selection: str, bindings, unbound):
    if selection == "all":
        return list(bindings), list(unbound)
    exact = [b for b in bindings if b.case_id == selection]
    if exact:
        return exact, []
    family = [b for b in bindings if b.family == selection or
              b.case_id.startswith(selection + ":") or
              b.case_id.startswith(selection + "-")]
    if family:
        return family, []
    unbound_exact = [u for u in unbound if u.case_id == selection]
    if unbound_exact:
        return [], unbound_exact
    known = [b.case_id for b in bindings] + [u.case_id for u in unbound]
    raise SystemExit(f"unknown case {selection!r}; known cases: {known}")
