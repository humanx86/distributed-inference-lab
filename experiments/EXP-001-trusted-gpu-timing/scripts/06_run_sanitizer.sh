#!/usr/bin/env bash
set -euo pipefail

######################################################
# STEP 1: Script preflight and fresh BUILD directory #
######################################################

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 BUILD_ID TOOL RUN_ID" >&2
    echo "Example: $0 BUILD-005-gpu-lab memcheck RUN-011-sanitizer-memcheck" >&2
    exit 2
fi

BUILD_ID="$1"
TOOL="$2"
RUN_ID="$3"

case "$TOOL" in
    memcheck|racecheck|synccheck)
        ;;
    *)
        echo "Unsupported sanitizer tool: $TOOL" >&2
        echo "Expected one of: memcheck, racecheck, synccheck" >&2
        exit 2
        ;;
esac

REPO="$(git rev-parse --show-toplevel)"
LAB="$REPO/experiments/EXP-001-trusted-gpu-timing"

BUILD_DIR="$REPO/build/EXP-001-trusted-gpu-timing/$BUILD_ID"
BINARY="$BUILD_DIR/gpu_lab"

RUN_ROOT="$LAB/runs"
RUN_DIR="$RUN_ROOT/$RUN_ID"

SANITIZER="$(command -v compute-sanitizer || true)"

if [[ -z "$SANITIZER" ]]; then
    echo "compute-sanitizer not found on PATH" >&2
    exit 2
fi

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

echo "Preflight passed:"
echo "  build: $BUILD_ID"
echo "  binary: $BINARY"
echo "  binary_sha256: $ACTUAL_BINARY_HASH"
echo "  sanitizer: $SANITIZER"
echo "  tool: $TOOL"
echo "  run: $RUN_ID"

##############################################
# STEP 3: Define the exact sanitizer command #
##############################################

SOURCE_COMMIT_FILE="$BUILD_DIR/source_commit.txt"

if [[ ! -f "$SOURCE_COMMIT_FILE" ]]; then
    echo "Source commit record not found: $SOURCE_COMMIT_FILE" >&2
    exit 2
fi

SOURCE_COMMIT="$(cat "$SOURCE_COMMIT_FILE")"

CMD=(
    "$SANITIZER"
    "--tool"
    "$TOOL"
    "--error-exitcode"
    "3"
    "$BINARY"
    "--check-only"
)

echo "Planned sanitizer execution:"
printf '  '
printf '%q ' "${CMD[@]}"
printf '\n'

echo "  source_commit: $SOURCE_COMMIT"
echo "  sanitizer error exit code: 3"
echo
echo "No sanitizer execution has occurred yet."

##########################
# STEP 3: create RUN-xxx #
##########################

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
    echo "sanitizer=$SANITIZER"
    echo "tool=$TOOL"
    echo "error_exitcode=3"
    "$SANITIZER" --version
} > "$RUN_DIR/sanitizer_identity.txt" 2>&1

STDOUT="$RUN_DIR/stdout.txt"
STDERR="$RUN_DIR/stderr.txt"
EXIT_CODE="$RUN_DIR/exit_code.txt"
CLASSIFICATION="$RUN_DIR/classification.txt"

cd "$BUILD_DIR"

set +e
"${CMD[@]}" > "$STDOUT" 2> "$STDERR"
RUN_STATUS=$?
set -e

printf '%s\n' "$RUN_STATUS" > "$EXIT_CODE"

case "$RUN_STATUS" in
    0)
        printf '%s\n' "CLEAN_COMPLETION" > "$CLASSIFICATION"
        ;;
    3)
        printf '%s\n' "SANITIZER_FINDING" > "$CLASSIFICATION"
        ;;
    *)
        printf '%s\n' "OTHER_FAILURE" > "$CLASSIFICATION"
        ;;
esac

echo "Sanitizer execution completed."
echo "  tool: $TOOL"
echo "  exit status: $RUN_STATUS"
echo
echo "Inspect:"
echo "  $STDOUT"
echo "  $STDERR"
echo "  $EXIT_CODE"
echo "  $CLASSIFICATION"

exit "$RUN_STATUS"
