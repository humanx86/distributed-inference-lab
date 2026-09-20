#!/usr/bin/env bash
set -euo pipefail
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
nvcc --version
nvidia-smi
cat > "$work_dir/smoke.cu" <<'CUDA'
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
void check(cudaError_t e, const char* label) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(e));
    std::exit(1);
  }
}
__global__ void transform(const int* in, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 2 * in[i] + 1;
}
int main() {
  cudaDeviceProp p{};
  check(cudaGetDeviceProperties(&p, 0), "properties");
  std::printf("GPU: %s; compute capability: %d.%d\n", p.name, p.major, p.minor);
  for (int n : {1, 255, 256, 257, 100003}) {
    std::vector<int> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = i % 101 - 50;
    int *d_in = nullptr, *d_out = nullptr;
    size_t bytes = n * sizeof(int);
    check(cudaMalloc(&d_in, bytes), "allocate input");
    check(cudaMalloc(&d_out, bytes), "allocate output");
    check(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice), "H2D");
    transform<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    check(cudaGetLastError(), "launch");
    check(cudaDeviceSynchronize(), "complete");
    check(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost), "D2H");
    for (int i = 0; i < n; ++i) {
      if (output[i] != 2 * input[i] + 1) {
        std::fprintf(stderr, "FAIL n=%d i=%d\n", n, i);
        return 1;
      }
    }
    check(cudaFree(d_in), "free input");
    check(cudaFree(d_out), "free output");
    std::printf("PASS n=%d\n", n);
  }
  std::puts("PASS: native CUDA compilation, execution, and checked results");
}
CUDA
# Native cubin only: avoids requiring this driver's PTX JIT to understand
# a newer toolkit's PTX. This is intentionally a laptop specific smoke test
nvcc -O2 -lineinfo -gencode arch=compute_120,code=sm_120 \
  "$work_dir/smoke.cu" -o "$work_dir/smoke"
"$work_dir/smoke"
for tool in ncu nsys compute-sanitizer; do
  if command -v "$tool" >/dev/null 2>&1; then
    "$tool" --version
  else
    printf '%s: not on PATH; capture/access remains untested\n' "$tool"
  fi
done
