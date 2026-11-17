# EXP-001: Gather correctness and timing

A bounded study of one row-gather operation: establish correctness, check CUDA
memory accesses, then measure explicitly defined timing intervals

The operation is:

```text
output[i, j] = input[indices[i], j]
```

For input shape `(R, C)` and `K` indices, the output shape is `(K, C)`.
Selection order is preserved, repeated indices are allowed, and the output
owns storage independent of the input.

The API supports contiguous row-major C++ `float` storage and signed
`std::int64_t` indices.

| Invalid input | Expected exception |
|---|---|
| Zero rows, zero columns or an empty index list | `std::invalid_argument` |
| Storage length differs from `rows * cols` | `std::invalid_argument` |
| Negative index or index greater than or equal to `input.rows` | `std::out_of_range` |
| Input or output shape product cannot fit in `std::size_t` | `std::overflow_error` |

## Implementation

The [CPU reference](src/gather_rows_cpu.cpp) uses nested row/column loops.
The [CUDA implementation](src/gather_rows_cuda.cu) independently maps one
thread to each output element. The host wrapper validates inputs, manages
device buffers/transfers, and checks CUDA API, launch and execution errors.
The implementations do not share an indexing helper.

## Run the checks

Run from the repository root in the configured development container.

### CPU

```bash
bash scripts/test_gather_rows_cpu.sh normal
bash scripts/test_gather_rows_cpu.sh sanitized
```

The sanitized mode enables host AddressSanitizer and UndefinedBehaviorSanitizer.

### CUDA

```bash
bash scripts/test_gather_rows_cuda.sh all
```

`all` builds once with `-lineinfo`, then runs the same binary normally, under
memcheck and under initcheck. Replace `all` with `normal`, `memcheck` or
`initcheck` to run one mode.

Each runner writes a fresh log under this experiment's `results/` directory.
The CUDA runner records versions, runner/source/binary checksums, source
verification, commands and exit statuses. Build artifacts go under
`build/EXP-001-model-execution/`.

## Test coverage

| Case group | CPU cases | CUDA cases |
|---|---:|---:|
| Valid gathers | 3 | 3 |
| Output-storage independence | 1 | 0 |
| Negative and upper-bound indices | 2 | 2 |
| Zero rows, zero columns and empty indices | 3 | 3 |
| Storage shorter or longer than the declared shape | 2 | 2 |
| Input/output shape-product overflow | 2 | 2 |
| **Total** | **13** | **12** |

## Recorded evidence

| Run | UTC timestamp | Observed result |
|---|---|---|
| [Sanitized CPU checkpoint](results/gather_rows_cpu_test_sanitized_FpbFob.log) | 2026-11-15T14:36:41Z | 13 cases passed; no host sanitizer diagnostics reported; exit 0 |
| [Combined CUDA runner](results/gather_rows_cuda_all_8X5BI0.log) | 2026-11-17T11:44:52Z | Source verification passed; 12 cases passed in each mode; zero sanitizer errors; zero memcheck leaks; all stage exits and overall exit 0 |

Earlier separate runs are retained as supporting records:

- [Memcheck](results/gather_rows_cuda_memcheck_6yqVkX.log),
  2026-11-17T10:59:03Z: 12 cases passed, zero errors/leaks, exit 0.
- [Initcheck](results/gather_rows_cuda_initcheck_tqEGQc.log),
  2026-11-17T11:04:27Z: the same binary was verified; 12 cases passed,

### Environment recorded by these runs

| Item | Recorded value |
|---|---|
| CPU compiler | GCC 13.3.0 |
| GPU | NVIDIA GeForce RTX 5070 Ti|
| Compute capability | 12.0 |
| CUDA compiler | NVCC 13.4.59 |
| Compute Sanitizer | 2026.3.0.0, build 38637409 |
| CUDA build | C++17, `-arch=native`, `-lineinfo`, host `-Wall,-Wextra` |

Memcheck used `--leak-check full --padding 32 --error-exitcode 99`.
Initcheck used `--check-api-memory-access yes --error-exitcode 99`.

## Limits and remaining work

Passing tests and sanitizers support the exercised paths, not every possible
input. Overflow fixtures use huge dimensions with tiny vectors to test
rejection order; they do not test large allocations or byte-count overflow.
No isolated sanitizer negative control or injected CUDA failure has been tested.
There are no performance results yet.

The timing part remains open.