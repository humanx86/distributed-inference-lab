#!/usr/bin/env bash
# TINY GPU TENSOR CORRECTNESS CHECK

set -euo pipefail

python -m pip check

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Build CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable"

a = torch.tensor([1, -2, 3], dtype=torch.float32, device="cuda:0")
b = torch.tensor([4, 5, -6], dtype=torch.float32, device="cuda:0")
c = a + b

torch.cuda.synchronize(0)

expected = torch.tensor([5, 3, -3], dtype=torch.float32, device="cuda:0")
torch.testing.assert_close(c, expected, rtol=0, atol=0)

print("Device:", c.device)
print("Shape:", tuple(c.shape))
print("Dtype:", c.dtype)
print("Result:", c.cpu().tolist())
print("PASS: GPU tensor correctness")
PY
