#!/usr/bin/env bash
set -euo pipefail

######################################################
# STEP 1: Script preflight and fresh BUILD directory #
######################################################

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 BUILD_ID RUN_ID" >&2
    echo "Example: $0 BUILD-004-gpu-lab RUN-006-parser-invalid-n" >&2
    exit 2
fi

BUILD_ID="$1"
RUN_ID="$2"

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

if [[ -e "$RUN_DIR" ]]; then
    echo "RUN directory already exists: $RUN_DIR" >&2
    exit 2
fi

echo "Preflight passed:"
echo "  build: $BUILD_ID"
echo "  binary: $BINARY"
echo "  run: $RUN_ID"
echo "  expected parser result: exit 2 for --n 1e6"

##########################################################
# STEP 2: Bind the negative-test RUN to the exact binary #
##########################################################

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

CMD=(
    "$BINARY"
    "--n"
    "1e6"
)

mkdir -p "$RUN_ROOT"
mkdir "$RUN_DIR"

{
    printf 'command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$RUN_DIR/command.txt"

printf '%s\n' "$BUILD_DIR" > "$RUN_DIR/working_directory.txt"

SOURCE_COMMIT="$(cat "$BUILD_DIR/source_commit.txt")"

{
    echo "build_id=$BUILD_ID"
    echo "build_directory=$BUILD_DIR"
    echo "binary=$BINARY"
    echo "binary_sha256=$ACTUAL_BINARY_HASH"
    echo "source_commit=$SOURCE_COMMIT"
    echo "expected_exit_code=2"
    echo "expected_classification=EXPECTED_NEGATIVE_TEST"
} > "$RUN_DIR/build_identity.txt"

echo "Prepared retained negative-test RUN:"
echo "  run: $RUN_ID"
echo "  build: $BUILD_ID"
echo "  command: gpu_lab --n 1e6"
echo "  expected exit: 2"
echo
echo "Prepared retained negative-test RUN metadata."

##########################################################
# STEP 3: Execute and classify the expected rejection    #
##########################################################

cd "$BUILD_DIR"

STDOUT="$RUN_DIR/stdout.txt"
STDERR="$RUN_DIR/stderr.txt"
EXIT_CODE="$RUN_DIR/exit_code.txt"
CLASSIFICATION="$RUN_DIR/classification.txt"

set +e
"${CMD[@]}" > "$STDOUT" 2> "$STDERR"
PROGRAM_STATUS=$?
set -e

printf '%s\n' "$PROGRAM_STATUS" > "$EXIT_CODE"

if [[ "$PROGRAM_STATUS" -eq 2 ]]; then
    printf '%s\n' "EXPECTED_NEGATIVE_TEST" > "$CLASSIFICATION"
    SCRIPT_STATUS=0
else
    printf '%s\n' "UNEXPECTED_RESULT" > "$CLASSIFICATION"
    SCRIPT_STATUS=1
fi

echo "Parser negative test completed."
echo "  observed program exit: $PROGRAM_STATUS"
echo "  expected program exit: 2"
echo
echo "Inspect:"
echo "  $STDOUT"
echo "  $STDERR"
echo "  $EXIT_CODE"
echo "  $CLASSIFICATION"

exit "$SCRIPT_STATUS"