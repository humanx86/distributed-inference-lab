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

## Next
Adding timers
