#!/usr/bin/env bash
# Build once; run each row-order comparison in a fresh process
set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: bash scripts/bench_gather_rows_order.sh" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/.."
experiment=experiments/EXP-001-model-execution
build_root=build/EXP-001-model-execution
mkdir -p -- "$experiment/results" "$build_root"
results=$(mktemp -d -- "$experiment/results/gather_timing_XXXXXX")
build="$build_root/${results##*/}"
mkdir -- "$build"
binary="$build/gather_rows_order"
printf 'Results: %s\nBuild and raw GPU snapshots: %s\n' "$results" "$build"

exec 3>"$results/run.log"
large_started=false
snapshot() {
  local label=$1 status=0
  timeout --kill-after=5s 10s nvidia-smi -q > "$build/gpu_${label}.txt" 2>&1 || status=$?
  printf 'gpu_%s_snapshot_exit=%s\n' "$label" "$status" >&3
}

finish() {
  local status=$?
  if [[ $large_started == true ]]; then snapshot after; fi
  printf 'run_exit=%s\n' "$status" >&3
  if (( status != 0 )); then
    printf 'Run stopped (exit %s). See %s/run.log and the log for the last command.\n' "$status" "$results" >&2
  fi
}
trap finish EXIT

run_step() {
  local limit=$1 status=0
  shift
  printf 'command=' >&3
  printf ' %q' "$@" >&3
  printf '\n' >&3
  timeout --kill-after=5s "$limit" "$@" || status=$?
  printf 'command_exit=%s\n' "$status" >&3
  return "$status"
}

date -u '+utc=%Y-%m-%dT%H:%M:%SZ' >&3
run_step 10s nvcc --version >&3
sources=("$experiment/src/gather_rows_cpu.cpp" "$experiment/src/gather_rows_cuda.cu"
         "$experiment/tests/gather_rows_order.cpp")
sha256sum -- "scripts/${BASH_SOURCE[0]##*/}" "$experiment/src/gather_rows_cpu.hpp" \
  "$experiment/src/gather_rows_cuda.hpp" "${sources[@]}" > "$results/sources.sha256"
run_step 120s nvcc -std=c++17 -O2 -arch=native -lineinfo -Xcompiler=-Wall,-Wextra \
  -I"$experiment/src" "${sources[@]}" -o "$binary" > "$build/compile.log" 2>&1
sha256sum --check "$results/sources.sha256" >&3
sha256sum -- "$binary" > "$results/binary.sha256"
run_step 60s "$binary" --plan > "$results/plan.csv.partial" 2> "$results/plan.log"
mv -- "$results/plan.csv.partial" "$results/plan.csv"

run_pair() {
  local name="${1}_cols${2}_${3}"
  local csv="$results/$name.csv"
  printf 'Running %s\n' "$name"
  run_step 60s "$binary" "$@" > "$csv.partial" 2> "$results/$name.log"
  # Two arms, each with 5 warm-up and 20 measured rows, plus the header
  if (( $(wc -l < "$csv.partial") != 51 )); then
    printf 'Incomplete or unexpected CSV: %s.partial (expected 51 lines)\n' "$csv" >&2
    return 1
  fi
  mv -- "$csv.partial" "$csv"
}

run_pair small 1 seq-first
run_pair small 1 shuffle-first
snapshot before
large_started=true
run_pair large 1 seq-first
run_pair large 32 shuffle-first
run_pair large 32 seq-first
run_pair large 1 shuffle-first