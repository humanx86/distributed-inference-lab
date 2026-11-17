#!/usr/bin/env bash
set -uo pipefail

# all: build once, then normal -> memcheck -> initcheck; stop at the first failure
# Time limits: build 120s, normal 30s, each sanitizer 120s, plus 5s kill grace

usage() {
  printf 'Usage: %s [normal|memcheck|initcheck|all]\nDefault: all\n' "$0"
}

if (( $# == 1 )) && [[ "$1" == --help || "$1" == -h ]]; then
  usage
  exit 0
fi
mode=${1:-all}
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
case "$mode" in
  normal|memcheck|initcheck) checks=("$mode") ;;
  all) checks=(normal memcheck initcheck) ;;
  *) usage >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
script_file="$script_dir/${BASH_SOURCE[0]##*/}"
repo_root=$(cd -- "$script_dir/.." && pwd) || exit 1
cd -- "$repo_root" || exit 1

experiment=experiments/EXP-001-model-execution
source_files=(
  "$experiment/src/gather_rows_cpu.hpp"
  "$experiment/src/gather_rows_cpu.cpp"
  "$experiment/src/gather_rows_cuda.hpp"
  "$experiment/src/gather_rows_cuda.cu"
  "$experiment/tests/gather_rows_cuda_test.cpp"
)
for source_file in "${source_files[@]}"; do
  if [[ ! -f "$source_file" ]]; then
    printf 'Missing source: %s\nPlace this runner in the repository scripts/ directory.\n' \
      "$source_file" >&2
    exit 1
  fi
done

results_dir="$experiment/results"
build_root=build/EXP-001-model-execution
mkdir -p -- "$results_dir" "$build_root" || exit 1
log_file=$(mktemp --suffix=.log \
  "$results_dir/gather_rows_cuda_${mode}_XXXXXX") || exit 1
# Separate build directories to prevent concurrent runs from replacing each other's binary
run_name=${log_file##*/}
build_dir="$build_root/${run_name%.log}"
mkdir -- "$build_dir" || exit 1
binary="$build_dir/gather_rows_cuda_test"

compile_cmd=(
  nvcc -std=c++17 -arch=native -lineinfo -Xcompiler=-Wall,-Wextra
  "$experiment/src/gather_rows_cpu.cpp"
  "$experiment/src/gather_rows_cuda.cu"
  "$experiment/tests/gather_rows_cuda_test.cpp"
  -o "$binary"
)

run_step() {
  local label=$1 seconds=$2 status
  shift 2
  printf '%s_command=' "$label" || return 1
  printf '%q ' "$@" || return 1
  printf '\n' || return 1
  timeout --kill-after=5s "${seconds}s" "$@"
  status=$?
  printf '%s_exit=%s\n' "$label" "$status" || return 1
  return "$status"
}

run_suite() {
  local source_identity check
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ' || return $?
  printf 'mode=%s\n' "$mode" || return 1
  timeout --kill-after=5s 10s nvcc --version || return $?
  if [[ "$mode" != normal ]]; then
    timeout --kill-after=5s 10s compute-sanitizer --version || return $?
    printf 'NV_COMPUTE_SANITIZER_BINARY_PATCHING=%s\n' \
      "${NV_COMPUTE_SANITIZER_BINARY_PATCHING-<unset>}" || return 1
  fi

  source_identity=$(sha256sum -- "$script_file" "${source_files[@]}") || return $?
  printf '%s\n' "$source_identity" || return 1
  run_step compile 120 "${compile_cmd[@]}" || return $?
  # Check that the recorded source identities still match after compilation
  printf 'source_verification_after_build:\n' || return 1
  sha256sum --check <<< "$source_identity" || return $?
  sha256sum -- "$binary" || return $?

  for check in "${checks[@]}"; do
    case "$check" in
      normal)
        run_step normal 30 "$binary" || return $?
        ;;
      memcheck)
        run_step memcheck 120 compute-sanitizer --tool memcheck \
          --leak-check full --padding 32 --error-exitcode 99 \
          "$binary" || return $?
        ;;
      initcheck)
        run_step initcheck 120 compute-sanitizer --tool initcheck \
          --check-api-memory-access yes --error-exitcode 99 \
          "$binary" || return $?
        ;;
    esac
  done
}

# Preserve the suite's status through tee. A logging failure also fails the run
run_suite 2>&1 | tee "$log_file"
pipeline_status=("${PIPESTATUS[@]}")
run_status=${pipeline_status[0]}
if (( run_status == 0 && pipeline_status[1] != 0 )); then
  run_status=${pipeline_status[1]}
fi

if ! printf 'build_and_test_exit=%s\n' "$run_status" >> "$log_file"; then
  printf 'FAIL: could not save the result footer.\n' >&2
  if (( run_status == 0 )); then run_status=1; fi
fi
printf 'build_and_test_exit=%s\n' "$run_status"
printf 'Log: %s/%s\n' "$repo_root" "$log_file"
exit "$run_status"
