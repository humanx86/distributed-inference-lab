#!/usr/bin/env bash
set -euo pipefail

# It supports normal and sanitized modes, records the compiler and source hashes, 
# saves a fresh log, and preserves failure exit codes through tee
# For sanitizer check:
# bash scripts/test_gather_rows_cpu.sh sanitized
# For ordinary checks:
# bash scripts/test_gather_rows_cpu.sh normal
mode=${1:-normal}
if (( $# > 1 )) || [[ "$mode" != normal && "$mode" != sanitized ]]; then
  printf 'Usage: %s [normal|sanitized]\n' "$0" >&2
  exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"

experiment=experiments/EXP-001-model-execution
build_dir=build/EXP-001-model-execution
results_dir="$experiment/results"
mkdir -p -- "$build_dir" "$results_dir"
binary="$build_dir/gather_rows_cpu_test_$mode"
log_file=$(mktemp --suffix=.log "$results_dir/gather_rows_cpu_test_${mode}_XXXXXX")

flags=(-std=c++17 -Wall -Wextra -Wconversion -Wsign-conversion -pedantic)
if [[ "$mode" == sanitized ]]; then
  flags+=(-O1 -g -fsanitize=address,undefined
          -fno-sanitize-recover=all -fno-omit-frame-pointer)
fi

compile_cmd=(g++ "${flags[@]}"
  "$experiment/src/gather_rows_cpu.cpp"
  "$experiment/tests/gather_rows_cpu_test.cpp"
  -o "$binary")

# Capture failure despite tee, then return it after writing the result footer
run_status=0
{
  printf 'utc=%s\nmode=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" &&
  printf 'ASAN_OPTIONS=%s\nUBSAN_OPTIONS=%s\n' \
    "${ASAN_OPTIONS-<unset>}" "${UBSAN_OPTIONS-<unset>}" &&
  g++ --version &&
  sha256sum "$experiment/src/gather_rows_cpu.hpp" \
    "$experiment/src/gather_rows_cpu.cpp" \
    "$experiment/tests/gather_rows_cpu_test.cpp" &&
  printf 'compile_command=' &&
  printf '%q ' "${compile_cmd[@]}" &&
  printf '\n' &&
  "${compile_cmd[@]}" &&
  "$binary"
} 2>&1 | tee "$log_file" || run_status=$?

printf 'build_and_test_exit=%s\n' "$run_status" | tee -a "$log_file"
printf 'Log: %s/%s\n' "$repo_root" "$log_file"
exit "$run_status"
