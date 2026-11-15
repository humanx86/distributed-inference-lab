# EXP-001: Gather correctness and timing

Current checkpoint: a CPU gather reference and 13 correctness cases

The operation is `output[i,j] = input[indices[i],j]`.
The API uses contiguous row-major float storage, signed 64-bit indices
and independently owned output storage. Empty inputs, invalid indices,
inconsistent storage and overflowing size products are rejected.

## Run CPU checks

From the repository root:

```bash
bash scripts/test_gather_rows_cpu.sh normal
bash scripts/test_gather_rows_cpu.sh sanitized
```

The runner records compiler identity, source hashes, the build command and exit status in a fresh log.

## Recorded evidence

[Sanitized CPU checkpoint](results/gather_rows_cpu_test_sanitized_FpbFob.log):
2026-11-15T14:36:41Z, GCC 13.3.0, all 13 cases passed, no sanitizer
diagnostics reported, exit status 0.

CUDA implementation, CUDA sanitizer checks and timing are next