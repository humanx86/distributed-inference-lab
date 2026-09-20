# Distributed Inference Lab

A working lab for learning and researching distributed inference systems, with an eventual focus on cross-layer performance diagnosis and sparse workloads, especially Mixture-of-Experts.

The lab keeps executable code, experiment records, measurements, and technical interpretation together. Results are scoped to the hardware, software, workloads, and timing boundaries actually tested.

## Current focus

### Phase I · Distributed Inference Foundations

**Lab 1: Trustworthy GPU Measurements**

**Lab question**

*What does each timer include, and does the execution trace support that interpretation?*

**Experiment:** [`EXP-001-trusted-gpu-timing`](experiments/EXP-001-trusted-gpu-timing/)

## Development environment

Open this repository's root folder in VS Code and select the CUDA Dev Container configuration under `.devcontainer/cuda/`.

See the [CUDA environment guide](containers/cuda/README.md) for setup, rootless Docker behavior, smoke checks, and known limitations. The image definition lives in the [Dockerfile](containers/cuda/Dockerfile), and Python development dependencies are listed in [requirements-dev.txt](requirements-dev.txt).

## Where work lives

| Path | Purpose |
| --- | --- |
| `.devcontainer/cuda/` | VS Code development configuration |
| `containers/cuda/` | CUDA image, environment guide, and smoke checks |
| `requirements-dev.txt` | Python development dependencies |
| `experiments/` | Experiment code, checks, analysis, and findings |
| `build/` | Generated executables when needed; excluded from Git |

## Evidence and results

Each experiment README records its purpose (learning, exploration, or confirmation) and links to its supporting evidence. Recorded executions receive distinct `RUN` identifiers, with outputs kept under the experiment's `runs/` directory.

For retained measurements, the command, working directory, executable source state, environment identity, inputs, correctness checks, timing boundary, raw sample order, and execution status are preserved.