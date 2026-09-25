#!/usr/bin/env bash
set -euo pipefail

########################################################
# STEP 1: Validate the requested timing case and build #
########################################################

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 BUILD_ID CASE RUN_ID" >&2
    echo "Cases: add-small add-large add-batch transform reduce add-small-repeat" >&2
    exit 2
fi

BUILD_ID="$1"
CASE="$2"
RUN_ID="$3"

case "$CASE" in
    add-small)
        OP="add"
        N="1003"
        BATCH="1"
        VARIANT=""
        ;;
    add-large)
        OP="add"
        N="1048576"
        BATCH="1"
        VARIANT=""
        ;;
    add-batch)
        OP="add"
        N="1003"
        BATCH="50"
        VARIANT=""
        ;;
    transform)
        OP="transform"
        N="1003"
        BATCH="1"
        VARIANT=""
        ;;
    reduce)
        OP="reduce"
        N="1003"
        BATCH="1"
        VARIANT="tree"
        ;;
    add-small-repeat)
        OP="add"
        N="1003"
        BATCH="1"
        VARIANT=""
        ;;
    *)
        echo "Unknown timing case: $CASE" >&2
        echo "Expected one of:" >&2
        echo "  add-small" >&2
        echo "  add-large" >&2
        echo "  add-batch" >&2
        echo "  transform" >&2
        echo "  reduce" >&2
        echo "  add-small-repeat" >&2
        exit 2
        ;;
esac

BLOCK="256"
WARMUP="5"
SAMPLES="8"
DATASET="dyadic"
GAP_US="0"
ATOL="1e-4"
RTOL="1e-4"

REPO="$(git rev-parse --show-toplevel)"
LAB="$REPO/experiments/EXP-001-trusted-gpu-timing"

BUILD_DIR="$REPO/build/EXP-001-trusted-gpu-timing/$BUILD_ID"
BINARY="$BUILD_DIR/gpu_lab"

RUN_ROOT="$LAB/runs"
RUN_DIR="$RUN_ROOT/$RUN_ID"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Build directory not found: $BUILD_DIR" >&2
    exit 2
fi

if [[ ! -x "$BINARY" ]]; then
    echo "Harness executable not found or not executable: $BINARY" >&2
    exit 2
fi

HASH_FILE="$BUILD_DIR/sha256.txt"

if [[ ! -f "$HASH_FILE" ]]; then
    echo "Build hash record not found: $HASH_FILE" >&2
    exit 2
fi

EXPECTED_BINARY_HASH="$(
    awk '$2 == "gpu_lab" { print $1 }' "$HASH_FILE"
)"

if [[ -z "$EXPECTED_BINARY_HASH" ]]; then
    echo "Could not find gpu_lab hash in: $HASH_FILE" >&2
    exit 2
fi

ACTUAL_BINARY_HASH="$(sha256sum "$BINARY" | awk '{print $1}')"

if [[ "$ACTUAL_BINARY_HASH" != "$EXPECTED_BINARY_HASH" ]]; then
    echo "Retained binary hash mismatch" >&2
    echo "Expected: $EXPECTED_BINARY_HASH" >&2
    echo "Actual:   $ACTUAL_BINARY_HASH" >&2
    exit 2
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "RUN directory already exists: $RUN_DIR" >&2
    exit 2
fi

echo "Timing-case preflight passed:"
echo "  build: $BUILD_ID"
echo "  case: $CASE"
echo "  op: $OP"
echo "  n: $N"
echo "  batch: $BATCH"
echo "  block: $BLOCK"
echo "  warmup: $WARMUP"
echo "  samples: $SAMPLES"
echo "  dataset: $DATASET"
echo "  gap_us: $GAP_US"
echo "  run: $RUN_ID"

if [[ -n "$VARIANT" ]]; then
    echo "  variant: $VARIANT"
fi

########################################
# STEP 2: Prepare the exact timing RUN #
########################################

SOURCE_COMMIT_FILE="$BUILD_DIR/source_commit.txt"

if [[ ! -f "$SOURCE_COMMIT_FILE" ]]; then
    echo "Source commit record not found: $SOURCE_COMMIT_FILE" >&2
    exit 2
fi

SOURCE_COMMIT="$(cat "$SOURCE_COMMIT_FILE")"

CMD=(
    "$BINARY"
    "--op" "$OP"
    "--n" "$N"
    "--block" "$BLOCK"
    "--warmup" "$WARMUP"
    "--samples" "$SAMPLES"
    "--batch" "$BATCH"
    "--dataset" "$DATASET"
    "--gap-us" "$GAP_US"
    "--atol" "$ATOL"
    "--rtol" "$RTOL"
)

if [[ -n "$VARIANT" ]]; then
    CMD+=("--variant" "$VARIANT")
fi

mkdir -p "$RUN_ROOT"
mkdir "$RUN_DIR"

{
    printf 'command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$RUN_DIR/command.txt"

printf '%s\n' "$BUILD_DIR" > "$RUN_DIR/working_directory.txt"

{
    echo "build_id=$BUILD_ID"
    echo "build_directory=$BUILD_DIR"
    echo "binary=$BINARY"
    echo "binary_sha256=$ACTUAL_BINARY_HASH"
    echo "source_commit=$SOURCE_COMMIT"
} > "$RUN_DIR/build_identity.txt"

{
    echo "case=$CASE"
    echo "op=$OP"
    echo "n=$N"
    echo "variant=${VARIANT:-na}"
    echo "block=$BLOCK"
    echo "warmup=$WARMUP"
    echo "samples=$SAMPLES"
    echo "batch=$BATCH"
    echo "dataset=$DATASET"
    echo "gap_us=$GAP_US"
    echo "atol=$ATOL"
    echo "rtol=$RTOL"
    echo "expected_rows=$SAMPLES"
    echo "cache_policy=repeated same input; no eviction; warm/cache-resident possible"
} > "$RUN_DIR/timing_config.txt"

echo "Prepared retained timing RUN:"
echo "  run: $RUN_ID"
echo "  case: $CASE"
echo "  build: $BUILD_ID"
echo
echo "Prepared retained timing RUN metadata"

############################################################
# STEP 3: Execute one timing case and preserve the raw CSV #
############################################################

TIMINGS="$RUN_DIR/timings.csv"
STDERR="$RUN_DIR/stderr.txt"
EXIT_CODE="$RUN_DIR/exit_code.txt"
EXECUTION_STATUS="$RUN_DIR/execution_status.txt"

cd "$BUILD_DIR"

set +e
"${CMD[@]}" > "$TIMINGS" 2> "$STDERR"
RUN_STATUS=$?
set -e

printf '%s\n' "$RUN_STATUS" > "$EXIT_CODE"

if [[ "$RUN_STATUS" -eq 0 ]]; then
    printf '%s\n' "COMPLETE_UNVALIDATED" > "$EXECUTION_STATUS"
else
    printf '%s\n' "PROGRAM_FAILURE" > "$EXECUTION_STATUS"
fi

echo "Timing execution completed."
echo "  case: $CASE"
echo "  program exit: $RUN_STATUS"
echo
echo "Inspect:"
echo "  $TIMINGS"
echo "  $STDERR"
echo "  $EXIT_CODE"
echo "  $EXECUTION_STATUS"

exit "$RUN_STATUS"
