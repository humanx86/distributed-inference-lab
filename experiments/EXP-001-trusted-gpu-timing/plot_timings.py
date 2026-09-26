#!/usr/bin/env python3

import argparse
import csv
import math
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_config(path):
    config = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        key, sep, value = line.partition("=")

        if not sep:
            raise ValueError(
                f"Malformed timing config line: {line!r}"
            )

        config[key] = value

    return config


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot one validated Lab 1 timing RUN "
            "using whole-batch timing boundaries."
        )
    )
    parser.add_argument("run_id")
    args = parser.parse_args()

    lab = pathlib.Path(__file__).resolve().parent
    run_dir = lab / "runs" / args.run_id
    figure_dir = lab / "figures"

    if not run_dir.is_dir():
        parser.error(
            f"RUN directory not found: {run_dir}"
        )

    validation_file = run_dir / "validation_status.txt"
    timing_file = run_dir / "timings.csv"
    config_file = run_dir / "timing_config.txt"

    if not validation_file.is_file():
        parser.error(
            f"Missing validation record: {validation_file}"
        )

    validation_status = validation_file.read_text().strip()

    if validation_status != "VALIDATED":
        parser.error(
            f"RUN is not validated: "
            f"{args.run_id} status={validation_status!r}"
        )

    if not timing_file.is_file():
        parser.error(
            f"Missing timing CSV: {timing_file}"
        )

    if not config_file.is_file():
        parser.error(
            f"Missing timing configuration: {config_file}"
        )

    with timing_file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        parser.error(
            "Timing CSV contains no data rows"
        )

    required_columns = {
        "sample",
        "n",
        "op",
        "variant",
        "block",
        "warmup",
        "batch",
        "gap_us",
        "dataset",
        "event_span_ms",
        "host_submit_ms",
        "host_complete_ms",
        "event_ms_per_op",
        "max_abs_error",
    }

    missing_columns = required_columns - set(rows[0])

    if missing_columns:
        parser.error(
            "Timing CSV missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    try:
        config = read_config(config_file)
    except ValueError as e:
        parser.error(str(e))

    required_config = {
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
        "expected_rows",
        "cache_policy",
    }

    missing_config = required_config - set(config)

    if missing_config:
        parser.error(
            "Timing config missing required fields: "
            + ", ".join(sorted(missing_config))
        )

    try:
        expected_rows = int(config["expected_rows"])
        batch = int(config["batch"])
    except ValueError as e:
        parser.error(
            f"Invalid numeric timing configuration: {e}"
        )

    if len(rows) != expected_rows:
        parser.error(
            f"Expected {expected_rows} timing rows, "
            f"found {len(rows)}"
        )

    expected_samples = list(range(expected_rows))

    try:
        actual_samples = [
            int(row["sample"])
            for row in rows
        ]
    except ValueError as e:
        parser.error(
            f"Invalid sample ordinal: {e}"
        )

    if actual_samples != expected_samples:
        parser.error(
            f"Unexpected sample order: {actual_samples}"
        )

    expected_fields = {
        "n": config["n"],
        "op": config["op"],
        "variant": config["variant"],
        "block": config["block"],
        "warmup": config["warmup"],
        "batch": config["batch"],
        "gap_us": config["gap_us"],
        "dataset": config["dataset"],
    }

    for row_number, row in enumerate(rows):
        for field, expected in expected_fields.items():
            if row[field] != expected:
                parser.error(
                    f"Row {row_number}: "
                    f"{field}={row[field]!r}, "
                    f"expected {expected!r}"
                )

    for row_number, row in enumerate(rows):
        try:
            event_span = float(row["event_span_ms"])
            host_submit = float(row["host_submit_ms"])
            host_complete = float(row["host_complete_ms"])
            event_per_op = float(row["event_ms_per_op"])
            error = float(row["max_abs_error"])
        except ValueError as e:
            parser.error(
                f"Row {row_number}: "
                f"invalid numeric field: {e}"
            )

        for name, value in {
            "event_span_ms": event_span,
            "host_submit_ms": host_submit,
            "host_complete_ms": host_complete,
            "event_ms_per_op": event_per_op,
        }.items():
            if not math.isfinite(value) or value <= 0:
                parser.error(
                    f"Row {row_number}: "
                    f"invalid {name}={value}"
                )

        if not math.isfinite(error) or error < 0:
            parser.error(
                f"Row {row_number}: "
                f"invalid max_abs_error={error}"
            )

        expected_per_op = event_span / batch

        if not math.isclose(
            event_per_op,
            expected_per_op,
            rel_tol=1e-7,
            abs_tol=0.0,
        ):
            parser.error(
                f"Row {row_number}: "
                "event_ms_per_op does not match "
                "event_span_ms / batch"
            )

    samples = [
        int(row["sample"])
        for row in rows
    ]

    event_span_ms = [
        float(row["event_span_ms"])
        for row in rows
    ]

    host_submit_ms = [
        float(row["host_submit_ms"])
        for row in rows
    ]

    host_complete_ms = [
        float(row["host_complete_ms"])
        for row in rows
    ]

    print(f"RUN: {args.run_id}")
    print(f"case: {config['case']}")
    print(f"rows: {len(rows)}")
    print(f"batch: {config['batch']}")
    print(f"validation: {validation_status}")
    print(
        f"cache policy: {config['cache_policy']}"
    )
    print("analysis preflight: PASS")
    print(f"source: {timing_file}")

    print()
    print("whole-batch timing series (ms):")

    for i, event, submit, complete in zip(
        samples,
        event_span_ms,
        host_submit_ms,
        host_complete_ms,
    ):
        print(
            f"sample={i} "
            f"event={event:.9f} "
            f"submit={submit:.9f} "
            f"complete={complete:.9f}"
        )

    figure_dir.mkdir(exist_ok=True)

    output = figure_dir / f"{args.run_id}.svg"

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2),
        layout="constrained",
    )

    ax.plot(
        samples,
        event_span_ms,
        marker="o",
        label="CUDA event span",
    )

    ax.plot(
        samples,
        host_submit_ms,
        marker="o",
        label="Host submission span",
    )

    ax.plot(
        samples,
        host_complete_ms,
        marker="o",
        label="Completed host span",
    )

    title = (
        f"{config['case']} — whole-batch timing boundaries\n"
        f"{args.run_id} · N={config['n']} · batch={config['batch']} · "
        f"block={config['block']} · warmup={config['warmup']}"
    )

    ax.set_title(title)
    ax.set_xlabel("Chronological sample ordinal")
    ax.set_ylabel("Whole-batch interval (ms)")
    ax.set_xticks(samples)
    ax.grid()
    ax.legend()

    fig.savefig(output)
    plt.close(fig)

    print()
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
