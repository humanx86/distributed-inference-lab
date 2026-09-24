#!/usr/bin/env bash
set -euo pipefail

######################################################
# STEP 1: Script preflight and fresh BUILD directory #
######################################################

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 BUILD_ID" >&2
    echo "Example: $0 BUILD-002-gpu-lab" >&2
    exit 2
fi

BUILD_ID="$1"

REPO="$(git rev-parse --show-toplevel)"
LAB="$REPO/experiments/EXP-001-trusted-gpu-timing"
SOURCE_REL="experiments/EXP-001-trusted-gpu-timing/gpu_lab.cu"
SOURCE="$REPO/$SOURCE_REL"

BUILD_ROOT="$REPO/build/EXP-001-trusted-gpu-timing"
BUILD_DIR="$BUILD_ROOT/$BUILD_ID"

if [[ ! -f "$SOURCE" ]]; then
    echo "Source not found: $SOURCE" >&2
    exit 2
fi

if ! git -C "$REPO" ls-files --error-unmatch "$SOURCE_REL" >/dev/null 2>&1; then
    echo "Source is not tracked by Git: $SOURCE_REL" >&2
    exit 2
fi

if ! git -C "$REPO" diff --quiet -- "$SOURCE_REL"; then
    echo "Source has unstaged changes: $SOURCE_REL" >&2
    exit 2
fi

if ! git -C "$REPO" diff --cached --quiet -- "$SOURCE_REL"; then
    echo "Source has staged but uncommitted changes: $SOURCE_REL" >&2
    exit 2
fi

if [[ -e "$BUILD_DIR" ]]; then
    echo "Build directory already exists: $BUILD_DIR" >&2
    exit 2
fi

mkdir -p "$BUILD_DIR"

echo "Created fresh build directory:"
echo "$BUILD_DIR"

################################################
# STEP 2: Retain and identify the exact source #
################################################

SOURCE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"

printf '%s\n' "$SOURCE_COMMIT" > "$BUILD_DIR/source_commit.txt"

cp "$SOURCE" "$BUILD_DIR/gpu_lab.cu"

SOURCE_HASH="$(sha256sum "$SOURCE" | awk '{print $1}')"
SNAPSHOT_HASH="$(sha256sum "$BUILD_DIR/gpu_lab.cu" | awk '{print $1}')"

{
    printf 'tracked_source_sha256=%s\n' "$SOURCE_HASH"
    printf 'retained_snapshot_sha256=%s\n' "$SNAPSHOT_HASH"
} > "$BUILD_DIR/source_identity.txt"

if [[ "$SOURCE_HASH" != "$SNAPSHOT_HASH" ]]; then
    echo "Retained source snapshot does not match tracked source" >&2
    exit 2
fi

echo "Retained source snapshot:"
echo "$BUILD_DIR/gpu_lab.cu"
echo "Source commit: $SOURCE_COMMIT"
echo "Source SHA256: $SOURCE_HASH"

#########################################
# STEP 3: Capture the build environment #
#########################################

NVCC="$(command -v nvcc || true)"

if [[ -z "$NVCC" ]]; then
    echo "nvcc not found on PATH" >&2
    exit 2
fi

if ! "$NVCC" --list-gpu-code | grep -qx 'sm_120'; then
    echo "nvcc does not report support for sm_120" >&2
    exit 2
fi

{
    echo "captured_at=$(date -Is)"
    echo "working_directory=$BUILD_DIR"
    echo "nvcc_path=$NVCC"
    echo

    nvidia-smi
    "$NVCC" --version
    python3 --version
    compute-sanitizer --version
    nsys --version

    echo
    echo "source_commit=$SOURCE_COMMIT"
    echo "git_status:"
    git -C "$REPO" status --short

    echo
    echo "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS-<unset>}"
    echo "NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS-<unset>}"
    echo "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING-<unset>}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"

    echo
    echo "supported_target:"
    "$NVCC" --list-gpu-code | grep -x 'sm_120'
} > "$BUILD_DIR/environment.txt" 2>&1

echo "Captured build environment:"
echo "$BUILD_DIR/environment.txt"

#################################################
# STEP 4: record and execute the retained build #
#################################################

BUILD_COMMAND=(
    "$NVCC"
    "-O3"
    "-lineinfo"
    "-std=c++17"
    "-arch=sm_120"
    "gpu_lab.cu"
    "-o"
    "gpu_lab"
)

{
    printf 'working_directory=%s\n' "$BUILD_DIR"
    printf 'command='
    printf '%q ' "${BUILD_COMMAND[@]}"
    printf '\n'
} > "$BUILD_DIR/build_command.txt"

cd "$BUILD_DIR"

set +e
"${BUILD_COMMAND[@]}" \
    > build.stdout.txt \
    2> build.stderr.txt
BUILD_STATUS=$?
set -e

printf '%s\n' "$BUILD_STATUS" > build.exit_code.txt

if [[ "$BUILD_STATUS" -ne 0 ]]; then
    echo "Build failed with exit status $BUILD_STATUS" >&2
    echo "Inspect:"
    echo "  $BUILD_DIR/build.stdout.txt"
    echo "  $BUILD_DIR/build.stderr.txt"
    exit "$BUILD_STATUS"
fi

if [[ ! -x "$BUILD_DIR/gpu_lab" ]]; then
    echo "Build reported success but executable is missing or not executable" >&2
    exit 2
fi

sha256sum gpu_lab.cu gpu_lab > sha256.txt

echo "Build completed successfully."
echo "Inspect:"
echo "  $BUILD_DIR/build_command.txt"
echo "  $BUILD_DIR/build.stdout.txt"
echo "  $BUILD_DIR/build.stderr.txt"
echo "  $BUILD_DIR/build.exit_code.txt"
echo "  $BUILD_DIR/sha256.txt"