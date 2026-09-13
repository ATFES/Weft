from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from kernel_configuration import PHYSICAL_OPTIONS, PROJECT, find_configuration


RUNNERS = {
    "kernel": "weft-kernel.sh",
    "mul-mat": "weft-mul-mat.sh",
    "vec-dot": "weft-quantized-vec-dot.sh",
    "row-dequantize": "weft-row-dequantize.sh",
}
SEARCH_ENV = {
    "lmul_eighths": "WEFT_TUNE_LMUL_EIGHTHS",
    "unroll": "WEFT_TUNE_UNROLLS",
    "pipeline_depth": "WEFT_TUNE_PIPELINE_DEPTHS",
    "scalar_load_prime": "WEFT_TUNE_SCALAR_LOAD_PRIMES",
    "partial_combine_policy": "WEFT_TUNE_PARTIAL_COMBINE_POLICIES",
    "record_axis_policy": "WEFT_TUNE_RECORD_AXIS_POLICIES",
}


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def source_candidates(base: dict[str, int]) -> list[dict[str, int]]:
    result = [dict(base)]
    spelling = os.environ.get("WEFT_TUNE_META_CHOICES")
    if not spelling:
        return result
    seen = set()
    for dimension in spelling.split(";"):
        name, separator, values = dimension.partition("=")
        if not separator or not name.isidentifier() or name in seen:
            raise ValueError(f"invalid source search dimension: {dimension}")
        seen.add(name)
        choices = list(dict.fromkeys(int(value) for value in values.split(",")))
        if not choices or min(choices) <= 0:
            raise ValueError(f"source choices must be positive: {dimension}")
        if len(result) * len(choices) > 1024:
            raise ValueError("source search exceeds the 1024-candidate hard limit")
        result = [dict(existing, **{name: value}) for existing in result for value in choices]
    return result


def parse_result(output: str) -> dict:
    values = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    if "configuration" not in values:
        raise ValueError("runner emitted no effective configuration")
    configuration = json.loads(values["configuration"])
    if values.get("numeric") not in ("bit-exact", "within-tolerance", "exact"):
        raise ValueError("runner emitted no successful numerical verdict")
    metrics = [name for name in ("cold_gop_s", "cold_melements_s", "melements_s") if name in values]
    if len(metrics) != 1:
        raise ValueError("runner must emit one throughput metric")
    metric = metrics[0]
    throughput = float(values[metric])
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("runner throughput must be finite and positive")
    shape = {name: int(values[name]) for name in ("M", "N", "K") if name in values}
    return {
        "configuration": configuration,
        "numeric": values["numeric"],
        "shape": shape,
        "input_policy": values.get("input_policy"),
        "metric": metric,
        "throughput": throughput,
        "measurements": {key: values[key] for key in (
            "repetitions", "cold_median_us", "cold_median_ms",
            "max_absolute_error", "max_relative_error") if key in values},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("sg2044", "k1", "v100"))
    parser.add_argument("kernel")
    parser.add_argument("repetitions", type=int)
    args = parser.parse_args()
    if args.repetitions < 10:
        raise ValueError("tuning requires at least ten real numerical/timing repetitions")
    runner = os.environ.get("WEFT_TUNE_RUNNER", "kernel")
    if runner not in RUNNERS:
        raise ValueError(f"unsupported tuning runner: {runner}")
    workload = args.kernel.split(":") if runner == "mul-mat" else [args.kernel]
    if runner == "mul-mat" and (len(workload) != 2 or workload[1] not in ("decode", "prefill")):
        raise ValueError("mul-mat tuning expects format:decode or format:prefill")
    request = " ".join((args.target, *workload))
    base = find_configuration(runner, request)
    sources = source_candidates(base["source"])
    domains = {}
    for name, environment in SEARCH_ENV.items():
        declared = base["physical_search"][name]
        spelling = os.environ.get(environment)
        values = spelling.split(",") if spelling else declared
        convert = PHYSICAL_OPTIONS[name][2]
        choices = list(dict.fromkeys(convert(value) for value in values))
        if not choices or any(value not in declared for value in choices):
            raise ValueError(f"{name} search must be a nonempty subset of {declared}")
        domains[name] = choices
    count = len(sources) * math.prod(len(values) for values in domains.values())
    budget = int(os.environ.get("WEFT_TUNE_MAX_CANDIDATES", base["search_budget"]))
    if not 1 <= budget <= 1024 or count > budget:
        raise ValueError(f"search has {count} candidates, budget {budget}; hard limit 1024")
    output = os.environ.get("WEFT_TUNE_OUT")
    if output:
        directory = Path(output).resolve()
        if directory.is_relative_to(PROJECT):
            raise ValueError("generated tuning artifacts must remain outside the repository")
        directory.mkdir(parents=True, exist_ok=False)
    else:
        directory = Path(tempfile.mkdtemp(prefix="weft-tune."))
    write_json(directory / "search.json", {
        "runner": runner, "request": request, "entry": base["entry"],
        "symbol": base["symbol"], "catalog": base["catalog"],
        "source_candidates": sources, "physical_domains": domains,
        "candidate_count": count, "budget": budget,
        "ranking": "measured-throughput-after-numerical-verification",
        "estimate": None, "repetitions": args.repetitions,
    })
    print(f"tuning_artifacts={directory}", flush=True)
    command = [str(PROJECT / "examples/run" / RUNNERS[runner]), args.target, *workload, str(args.repetitions)]
    selected = None
    comparison = None
    candidates = itertools.product(sources, itertools.product(*domains.values()))
    for number, (source, choices) in enumerate(candidates):
        physical = dict(zip(domains, choices))
        environment = dict(os.environ)
        environment.pop("WEFT_TUNE_SELECTION", None)
        environment["WEFT_KEEP_ARTIFACTS"] = "1"
        environment["WEFT_META_BINDINGS"] = ";".join(f"{key}={value}" for key, value in sorted(source.items()))
        for name, (variable, _, _) in PHYSICAL_OPTIONS.items():
            environment[variable] = str(physical[name])
        completed = subprocess.run(command, env=environment, capture_output=True, text=True)
        prefix = directory / f"candidate-{number:04d}"
        prefix.with_suffix(".stdout").write_text(completed.stdout)
        prefix.with_suffix(".stderr").write_text(completed.stderr)
        record = {"source": source, "physical": physical,
                  "exit_code": completed.returncode, "status": "rejected"}
        if completed.returncode:
            record["reason"] = completed.stderr or completed.stdout
        else:
            try:
                measured = parse_result(completed.stdout)
            except (ValueError, KeyError) as error:
                record["reason"] = str(error)
            else:
                identity = (measured["shape"], measured["input_policy"], measured["metric"],
                            measured["configuration"]["entry"], measured["configuration"]["target"])
                if comparison is not None and identity != comparison:
                    raise ValueError("candidate changed workload, numerical input policy, metric, entry, or target")
                comparison = identity
                record.update(measured, status="measured")
                if selected is None or measured["throughput"] > selected["throughput"]:
                    selected = dict(record, candidate=number)
        write_json(prefix.with_suffix(".json"), record)
        print(f"candidate={number} status={record['status']} physical={json.dumps(physical, sort_keys=True)}", flush=True)
        if record["status"] == "rejected":
            print(f"rejection={record['reason']}", flush=True)
    if selected is None:
        raise RuntimeError(f"no numerically valid measurable candidate; see {directory}")
    configuration = selected["configuration"]
    selection = {key: configuration[key] for key in ("runner", "request", "entry", "symbol", "target")}
    selection.update(source=selected["source"], physical=selected["physical"],
                     shape=selected["shape"], input_policy=selected["input_policy"],
                     numeric=selected["numeric"], metric=selected["metric"],
                     throughput=selected["throughput"], candidate=selected["candidate"],
                     candidate_count=count, search=str(directory / "search.json"))
    selection_path = directory / "selected.json"
    write_json(selection_path, selection)
    print(f"winner={json.dumps(selection, sort_keys=True)}", flush=True)
    print(f"selected_binding={selection_path}", flush=True)
    if os.environ.get("WEFT_TUNE_APPLY_WINNER", "1") == "1":
        environment = dict(os.environ)
        for variable, _, _ in PHYSICAL_OPTIONS.values():
            environment.pop(variable, None)
        environment.pop("WEFT_META_BINDINGS", None)
        environment["WEFT_TUNE_SELECTION"] = str(selection_path)
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
