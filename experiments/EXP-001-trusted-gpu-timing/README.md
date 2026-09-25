# EXP-001: Trusted GPU Timing

## Mode
Learning

## Question
What exactly does each timing boundary include for a correct GPU operation?

## Scope
One GPU; vector operations first, followed by the supplied complete reduction.

## Current status
Vector-add implementation written using RAII.
Boundary-size correctness validation done.
Timing work has not started.

## Evidence
- `vector_add.cu`: personal vector-add implementation
- `gpu_lab.cu`: supplied teaching harness
- `cpu_models.py`: supplied CPU conceptual checks

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

## Next
Adding timers
