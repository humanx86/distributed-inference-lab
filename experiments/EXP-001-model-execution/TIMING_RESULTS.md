# CUDA row-gather timing

On a GeForce RTX 5070 Ti GPU, shuffling the rows of a 72 MiB input increased the one-column gather’s median device-event interval by a factor of **3.48–7.29**. At 32 columns the ratio was **0.95–0.99**. Both execution orders supported the prediction `R(1) > R(32)`, where `R` is the shuffled median divided by the sequential median in the same process.

## Comparison

The kernel copies `output[i, j] = input[indices[i], j]` from contiguous row-major floats using signed 64-bit indices. Both orders selected every row exactly once from the same input. The benchmark driver checked the output of each order against a CPU reference before emitting samples. Data and shuffle seeds were 17 and 42.

| Input | Columns | Order tested first | Sequential median (µs) | Shuffled median (µs) | R |
|---|---:|---|---:|---:|---:|
| 64 KiB | 1 | Sequential | 3.744 | 4.544 | 1.214 |
| 64 KiB | 1 | Shuffled | 4.000 | 4.656 | 1.164 |
| 72 MiB | 1 | Sequential | 738.016 | 2,571.184 | 3.484 |
| 72 MiB | 1 | Shuffled | 484.848 | 3,532.288 | 7.285 |
| 72 MiB | 32 | Sequential | 461.056 | 439.840 | 0.954 |
| 72 MiB | 32 | Shuffled | 460.896 | 456.720 | 0.991 |

Each row represents a separate process. Each ordering had five warm-up iterations and twenty measured iterations per timing pass. The table uses all twenty measured device-event samples per ordering. In the 72 MiB sequential-first process, the first measured shuffled iteration at width 1 took 3.708 ms. The next nineteen ranged from 2.568 to 2.574 ms. The change in timing after warm-up is part of the result.

## What the clocks measure

The device pass records CUDA events around the kernel on the same stream and waits for the stop event before reading elapsed time. A separate host pass reads a steady clock before launch, immediately after launch returns, and after the stream wait returns. The first two readings give host enqueue time. The first and third give synchronized host elapsed time, including the launch check and wait. Allocation and input/output transfers occur outside these intervals. Device and host entries on the same CSV row refer to different launches.

At width 1, shuffling can spread a warp’s input loads across many rows. At width 32, a warp reads adjacent floats within one selected row. On the large input, the index array also changes from 144 MiB at width 1 to 4.5 MiB at width 32. The within-width comparisons hold that size fixed. No cache-hit, transaction-count or DRAM-traffic measurements were collected.

## GPU conditions during the large-input series

| Reading | Before the four processes | After the four processes |
|---|---:|---:|
| Performance state | P4 | P0 |
| SM clock | 1,065 MHz | 1,732 MHz |
| Memory clock | 9,001 MHz | 14,001 MHz |
| GPU temperature | 40 °C | 41 °C |
| Instantaneous power draw | 14.74 W | 15.49 W |
| Current power limit | 95 W | 95 W |
| Software power cap flag | Active | Not active |
| GPU utilization | 0% | 0% |
| Processes listed | None | None |

The software power-capping and software thermal-slowdown counters each increased by 2.927 s between snapshots.

## Rerun

In the CUDA development container, run the timing script from the repository root

```bash
bash scripts/bench_gather_rows_order.sh
```

The runner builds once and runs the six paired comparisons in the order used for this study. Each invocation creates a fresh directory under `experiments/EXP-001-model-execution/results/` containing the workload plan, six sample CSVs, per-process logs, checksums and `run.log`