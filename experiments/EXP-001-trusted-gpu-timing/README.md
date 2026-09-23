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

## Next
Adding timers
