#!/usr/bin/env bash
set -euo pipefail

########################################
# Piece 1: rgument and build preflight #
########################################

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 BUILD_ID BLOCK RUN_ID" >&2
    echo "Example: $0 BUILD-004-gpu-lab 128 RUN-002-harness-block128" >&2
    exit 2
fi

BUILD_ID="$1"
BLOCK="$2"
RUN_ID="$3"

case "$BLOCK" in
    128|256|512)
        ;;
    *)
        echo "Unsupported block size: $BLOCK" >&2
        echo "Expected one of: 128, 256, 512" >&2
        exit 2
        ;;
esac

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
echo "  block: $BLOCK"
echo "  run: $RUN_ID"

#########################################################
# Piece 2: Bind a fresh RUN to the exact retained build #
#########################################################

# verify the retained binary has not changed
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

# define the exact command
CMD=(
    "$BINARY"
    "--check-only"
    "--block"
    "$BLOCK"
)

# create the fresh RUN directory
mkdir -p "$RUN_ROOT"
mkdir "$RUN_DIR"

# record the command
{
    printf 'command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$RUN_DIR/command.txt"

# record the working directory 
printf '%s\n' "$BUILD_DIR" > "$RUN_DIR/working_directory.txt"

# record the build identity
SOURCE_COMMIT="$(cat "$BUILD_DIR/source_commit.txt")"

{
    echo "build_id=$BUILD_ID"
    echo "build_directory=$BUILD_DIR"
    echo "binary=$BINARY"
    echo "binary_sha256=$ACTUAL_BINARY_HASH"
    echo "source_commit=$SOURCE_COMMIT"
} > "$RUN_DIR/build_identity.txt"

echo "Prepared retained RUN:"
echo "  run: $RUN_ID"
echo "  build: $BUILD_ID"
echo "  block: $BLOCK"
echo
echo "Prepared retained RUN metadata."

##################################################
# Piece 3: Execute exactly one correctness suite #
##################################################

cd "$BUILD_DIR"

STDOUT="$RUN_DIR/stdout.txt"
STDERR="$RUN_DIR/stderr.txt"
EXIT_CODE="$RUN_DIR/exit_code.txt"

set +e
"${CMD[@]}" > "$STDOUT" 2> "$STDERR"
RUN_STATUS=$?
set -e

printf '%s\n' "$RUN_STATUS" > "$EXIT_CODE"

echo "Harness execution completed."
echo "  exit status: $RUN_STATUS"
echo
echo "Inspect:"
echo "  $STDOUT"
echo "  $STDERR"
echo "  $EXIT_CODE"

exit "$RUN_STATUS"