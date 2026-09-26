#!/usr/bin/env python3

import argparse
import csv
import math
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_key_value_file(path):
    values = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        key, sep, value = line.partition("=")

        if not sep:
            raise ValueError(
                f"Malformed metadata line in {path}: {line!r}"
            )

        values[key] = value

    return values


def load_run(lab, run_id, parser):
    run_dir = lab / "runs" / run_id

    if not run_dir.is_dir():
        parser.error(
            f"RUN directory not found: {run_dir}"
        )

    validation_file = run_dir / "validation_status.txt"
    timing_file = run_dir / "timings.csv"
    config_file = run_dir / "timing_config.txt"
    build_file = run_dir / "build_identity.txt"

    if not validation_file.is_file():
        parser.error(
            f"Missing validation record: {validation_file}"
        )

    validation_status = validation_file.read_text().strip()

    if validation_status != "VALIDATED":
        parser.error(
            f"RUN is not validated: "
            f"{run_id} status={validation_status!r}"
        )

    if not timing_file.is_file():
        parser.error(
            f"Missing timing CSV: {timing_file}"
        )

    if not config_file.is_file():
        parser.error(
            f"Missing timing configuration: {config_file}"
        )

    if not build_file.is_file():
        parser.error(
            f"Missing build identity: {build_file}"
        )

    try:
        config = read_key_value_file(config_file)
        build = read_key_value_file(build_file)
    except ValueError as e:
        parser.error(str(e))

    with timing_file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        parser.error(
            f"Timing CSV has no data rows: {timing_file}"
        )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "config": config,
        "build": build,
        "rows": rows,
    }


def require_same_config(a, b, parser):
    required_fields = {
        "case",
        "op",
        "n",
        "variant",
        "block",
        "warmup",
        "samples",
        "batch",
        "dataset",
        "gap_us",
        "atol",
        "rtol",
        "expected_rows",
        "cache_policy",
    }

    for run in (a, b):
        missing = required_fields - set(run["config"])

        if missing:
            parser.error(
                f"{run['run_id']}: timing config missing fields: "
                + ", ".join(sorted(missing))
            )

    # "case" is intentionally excluded here because it describes
    # the experimental role: add-small vs add-batch.
    #
    # "batch" is intentionally excluded because it is the controlled
    # variable this comparison is testing.
    same_fields = [
        "op",
        "n",
        "variant",
        "block",
        "warmup",
        "samples",
        "dataset",
        "gap_us",
        "atol",
        "rtol",
        "expected_rows",
        "cache_policy",
    ]

    for field in same_fields:
        av = a["config"][field]
        bv = b["config"][field]

        if av != bv:
            parser.error(
                f"Incomparable RUNs: "
                f"{field} differs: {av!r} vs {bv!r}"
            )

    batch_a = a["config"]["batch"]
    batch_b = b["config"]["batch"]

    if batch_a == batch_b:
        parser.error(
            "Expected batching contrast, "
            f"but both RUNs use batch={batch_a}"
        )

    hash_a = a["build"].get("binary_sha256")
    hash_b = b["build"].get("binary_sha256")

    if not hash_a or not hash_b:
        parser.error(
            "Missing binary_sha256 in build identity"
        )

    if hash_a != hash_b:
        parser.error(
            "Incomparable RUNs: binary SHA256 differs"
        )


def normalize_run(run, parser):
    required_columns = {
        "sample",
        "batch",
        "event_span_ms",
        "host_submit_ms",
        "host_complete_ms",
        "event_ms_per_op",
    }

    rows = run["rows"]

    missing = required_columns - set(rows[0])

    if missing:
        parser.error(
            f"{run['run_id']}: missing columns: "
            + ", ".join(sorted(missing))
        )

    try:
        batch = int(run["config"]["batch"])
        expected_rows = int(run["config"]["expected_rows"])
    except (KeyError, ValueError) as e:
        parser.error(
            f"{run['run_id']}: invalid timing config: {e}"
        )

    if batch <= 0:
        parser.error(
            f"{run['run_id']}: batch must be positive"
        )

    if len(rows) != expected_rows:
        parser.error(
            f"{run['run_id']}: "
            f"expected {expected_rows} rows, "
            f"found {len(rows)}"
        )

    samples = []
    event_per_op = []
    submit_per_op = []
    complete_per_op = []

    for row_number, row in enumerate(rows):
        try:
            sample = int(row["sample"])
            row_batch = int(row["batch"])

            event = float(row["event_span_ms"])
            submit = float(row["host_submit_ms"])
            complete = float(row["host_complete_ms"])
            stored_event_per_op = float(row["event_ms_per_op"])
        except ValueError as e:
            parser.error(
                f"{run['run_id']} row {row_number}: "
                f"invalid numeric field: {e}"
            )

        if sample != row_number:
            parser.error(
                f"{run['run_id']}: "
                f"unexpected sample ordinal "
                f"{sample} at row {row_number}"
            )

        if row_batch != batch:
            parser.error(
                f"{run['run_id']} row {row_number}: "
                f"batch={row_batch}, expected {batch}"
            )

        numeric_values = {
            "event_span_ms": event,
            "host_submit_ms": submit,
            "host_complete_ms": complete,
            "event_ms_per_op": stored_event_per_op,
        }

        for name, value in numeric_values.items():
            if not math.isfinite(value) or value <= 0:
                parser.error(
                    f"{run['run_id']} row {row_number}: "
                    f"invalid {name}={value}"
                )

        derived_event_per_op = event / batch

        if not math.isclose(
            stored_event_per_op,
            derived_event_per_op,
            rel_tol=1e-7,
            abs_tol=0.0,
        ):
            parser.error(
                f"{run['run_id']} row {row_number}: "
                "stored event_ms_per_op does not match "
                "event_span_ms / batch"
            )

        samples.append(sample)
        event_per_op.append(
            derived_event_per_op
        )
        submit_per_op.append(
            submit / batch
        )
        complete_per_op.append(
            complete / batch
        )

    return {
        "samples": samples,
        "batch": batch,
        "event_per_op": event_per_op,
        "submit_per_op": submit_per_op,
        "complete_per_op": complete_per_op,
    }


def print_normalized(run, normalized):
    print()
    print(
        f"{run['run_id']} "
        f"(batch={normalized['batch']})"
    )

    print(
        "derived batch-average intervals "
        "per operation (ms):"
    )

    for sample, event, submit, complete in zip(
        normalized["samples"],
        normalized["event_per_op"],
        normalized["submit_per_op"],
        normalized["complete_per_op"],
    ):
        print(
            f"sample={sample} "
            f"event={event:.9f} "
            f"submit={submit:.9f} "
            f"complete={complete:.9f}"
        )


def summarize(normalized):
    return {
        "event": statistics.median(
            normalized["event_per_op"]
        ),
        "submit": statistics.median(
            normalized["submit_per_op"]
        ),
        "complete": statistics.median(
            normalized["complete_per_op"]
        ),
    }


def print_summary(run, normalized, summary):
    print()
    print(
        f"{run['run_id']} "
        f"(batch={normalized['batch']}) "
        "median batch-average intervals (ms/op):"
    )

    print(
        f"  CUDA event span:      "
        f"{summary['event']:.9f}"
    )

    print(
        f"  Host submission span: "
        f"{summary['submit']:.9f}"
    )

    print(
        f"  Completed host span:  "
        f"{summary['complete']:.9f}"
    )


def plot_comparison(
    lab,
    a,
    b,
    normalized_a,
    normalized_b,
    summary_a,
    summary_b,
):
    figure_dir = lab / "figures"
    figure_dir.mkdir(exist_ok=True)

    case_a = a["config"]["case"]
    case_b = b["config"]["case"]

    output = (
        figure_dir
        / f"batch-comparison-{case_a}-vs-{case_b}.svg"
    )

    categories = [
        "CUDA event",
        "Host submission",
        "Completed host",
    ]

    values_a = [
        summary_a["event"],
        summary_a["submit"],
        summary_a["complete"],
    ]

    values_b = [
        summary_b["event"],
        summary_b["submit"],
        summary_b["complete"],
    ]

    x = list(range(len(categories)))
    width = 0.36

    x_a = [
        value - width / 2
        for value in x
    ]

    x_b = [
        value + width / 2
        for value in x
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2),
        layout="constrained",
    )

    ax.bar(
        x_a,
        values_a,
        width=width,
        label=f"batch={normalized_a['batch']}",
    )

    ax.bar(
        x_b,
        values_b,
        width=width,
        label=f"batch={normalized_b['batch']}",
    )

    ax.set_title(
        "Normalized batch-average timing comparison\n"
        f"{a['run_id']} vs {b['run_id']} · "
        f"N={a['config']['n']} · "
        f"block={a['config']['block']}"
    )

    ax.set_xlabel("Timing boundary")
    ax.set_ylabel(
        "Median batch-average interval per operation (ms)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.legend()

    fig.savefig(output)
    plt.close(fig)

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two validated Lab 1 timing RUNs "
            "using derived batch-average intervals."
        )
    )

    parser.add_argument("run_a")
    parser.add_argument("run_b")

    args = parser.parse_args()

    lab = pathlib.Path(__file__).resolve().parent

    a = load_run(
        lab,
        args.run_a,
        parser,
    )

    b = load_run(
        lab,
        args.run_b,
        parser,
    )

    require_same_config(
        a,
        b,
        parser,
    )

    normalized_a = normalize_run(
        a,
        parser,
    )

    normalized_b = normalize_run(
        b,
        parser,
    )

    if normalized_a["samples"] != normalized_b["samples"]:
        parser.error(
            "Incomparable RUNs: "
            "sample ordinals differ"
        )

    print(f"RUN A: {a['run_id']}")
    print(
        f"  case: {a['config']['case']}"
    )
    print(
        f"  batch: {a['config']['batch']}"
    )
    print(
        f"  rows: {len(a['rows'])}"
    )

    print(f"RUN B: {b['run_id']}")
    print(
        f"  case: {b['config']['case']}"
    )
    print(
        f"  batch: {b['config']['batch']}"
    )
    print(
        f"  rows: {len(b['rows'])}"
    )

    print(
        "controlled difference: batch "
        f"{a['config']['batch']} "
        f"→ {b['config']['batch']}"
    )

    print(
        f"common binary SHA256: "
        f"{a['build']['binary_sha256']}"
    )

    print("comparability check: PASS")

    print_normalized(
        a,
        normalized_a,
    )

    print_normalized(
        b,
        normalized_b,
    )

    summary_a = summarize(
        normalized_a
    )

    summary_b = summarize(
        normalized_b
    )

    print_summary(
        a,
        normalized_a,
        summary_a,
    )

    print_summary(
        b,
        normalized_b,
        summary_b,
    )

    output = plot_comparison(
        lab,
        a,
        b,
        normalized_a,
        normalized_b,
        summary_a,
        summary_b,
    )

    print()
    print(
        "summary statistic: median across "
        "retained chronological samples"
    )

    print(
        "interpretation boundary: values are "
        "derived batch averages, not individually "
        "measured operation latencies"
    )

    print()
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
    