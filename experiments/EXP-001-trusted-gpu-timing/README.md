# EXP-001: Trusted GPU Timing

## Mode
Learning

## Question
What exactly does each timing boundary include for a correct GPU operation?

## Scope
One GPU; vector operations first, followed by the supplied complete reduction.

## Current status
Personal vector-add implementation written using RAII.

BUILD-005 harness correctness and targeted fixtures pass for the tested
configurations. Compute Sanitizer reports clean completion for the
retained memcheck, racecheck, and synccheck runs.

The timing matrix RUN-014 through RUN-019 has been captured and validated.

Plots and the normalized batching comparison
have been generated.

Next: the annotated Nsight Systems timeline

## Evidence
- `vector_add.cu`: personal vector-add implementation
- `gpu_lab.cu`: supplied teaching harness
- `cpu_models.py`: supplied CPU conceptual checks
- `plot_timings.py`: validated chronological timing analysis
- `plot_batch_comparison.py`: validated normalized batching comparison
- `runs/`: retained execution records and raw outputs
- `figures/`: retained timing-analysis figures

### RUN-001 — vector-add correctness

- Build: `BUILD-001-vector-add`
- Purpose: retained correctness check for the personal vector-add implementation
- Sizes: `1, 255, 256, 257, 1000, 1024`
- Result: PASS for all tested sizes
- Exit status: `0`
- Evidence: `runs/RUN-001-vector-add-correctness/`

## Harness correctness

Build: `BUILD-004-gpu-lab`

Retained correctness runs:

- `RUN-003-harness-block128` — PASS, exit 0
- `RUN-004-harness-block256` — PASS, exit 0
- `RUN-005-harness-block512` — PASS, exit 0

This supports correctness for the harness's tested sizes,
datasets, operations, and reduction variants under this retained build.

### Harness CLI validation

- `RUN-007-parser-invalid-n` — invalid scientific-notation count input rejected as expected, exit 2

### BUILD-005 harness correctness

- `RUN-008-harness-block128` — PASS
- `RUN-009-harness-block256` — PASS
- `RUN-010-harness-block512` — PASS

Each run passed the ordinary boundary suite and the targeted
last-valid-element fixture. The cancellation fixture produced the
expected order-sensitive observation:

`device=0`, `reference=1`, `abs_error=1`, `within_tolerance=no`.

### BUILD-005 sanitizer validation

Block-256 `--check-only` suite:

- `RUN-011-sanitizer-memcheck` — clean completion, 0 reported errors
- `RUN-012-sanitizer-racecheck` — clean completion, 0 hazards
- `RUN-013-sanitizer-synccheck` — clean completion, 0 reported errors

Compute Sanitizer: 2026.3.0.0.

These results cover only the executed paths, they do not
prove general memory safety, race freedom, synchronization correctness,
or timing validity.

## Timing runs

All timing runs below use the retained BUILD-005 `gpu_lab` executable.

| RUN | Case | Key configuration | Purpose | Status |
| --- | --- | --- | --- | --- |
| RUN-014-timing-add-small | add-small | N=1003, batch=1 | Small add baseline | VALIDATED |
| RUN-015-timing-add-large | add-large | N=1048576, batch=1 | Input-size comparison | VALIDATED |
| RUN-016-timing-add-batch | add-batch | N=1003, batch=50 | Batching comparison | VALIDATED |
| RUN-017-timing-transform | transform | N=1003, batch=1 | Operation comparison | VALIDATED |
| RUN-018-timing-reduce | reduce/tree | N=1003, batch=1 | Multi-stage reduction timing | VALIDATED |
| RUN-019-timing-add-small-repeat | add-small | N=1003, batch=1 | Fresh-process repeat | VALIDATED |

## Timing analysis

Chronological whole-batch plots:

- `figures/RUN-014-timing-add-small.svg`
- `figures/RUN-015-timing-add-large.svg`
- `figures/RUN-016-timing-add-batch.svg`
- `figures/RUN-017-timing-transform.svg`
- `figures/RUN-018-timing-reduce.svg`
- `figures/RUN-019-timing-add-small-repeat.svg`

Normalized batching comparison:

- `figures/batch-comparison-add-small-vs-add-batch.svg`

#### Observation
- the median whole-batch spans increase by a factor of less than 50 from batch 1 to batch 50;
- normalized medians become around 1.7–1.8 us/op at batch 50;
- host submission doesn't change much after normalization compared
  with the event and completed host boundaries;
- from the plots there is no sustained drift within any run

#### Interpretation
- submission scales close to operation count;
- event and completed-host boundaries scale more sublinearly;
- this is consistent with amortization and/or steady-state effects, but
  the timing data do not isolate the mechanism;

#### Limits
- 1.7 us/op is a derived batch average, not individual kernel latency;
- no latency distribution or tail claim;
- no claim that batching made each kernel ~5x faster;
- results apply to this retained workload/configuration;
- cache policy was repeated same input with no eviction, so warm/cache
  residency is possible;
