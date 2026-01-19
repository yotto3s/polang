#!/bin/bash
# Run clang-tidy on the Polang codebase
# Usage: ./scripts/run-clang-tidy.sh [build_dir] [--fix] [files...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

# Parse arguments
FIX_FLAG=""
FILES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_FLAG="--fix"
            shift
            ;;
        *)
            # First non-flag argument that is a directory is the build dir
            if [[ -d "$1" ]] && [[ ${#FILES[@]} -eq 0 ]] && [[ "$BUILD_DIR" == "$PROJECT_DIR/build" ]]; then
                BUILD_DIR="$1"
            else
                FILES+=("$1")
            fi
            shift
            ;;
    esac
done

# Check if build directory exists
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: Build directory '$BUILD_DIR' not found. Run cmake first."
    exit 1
fi

# Check if compile_commands.json exists
if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: compile_commands.json not found in '$BUILD_DIR'. Configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    exit 1
fi

# Create filtered compile_commands.json (remove GCC-specific flags)
# Create a temp directory with proper structure for clang-tidy
TEMP_BUILD_DIR=$(mktemp -d)
trap "rm -rf $TEMP_BUILD_DIR" EXIT

sed -e 's/-fno-lifetime-dse//g' \
    -e 's/-Wno-class-memaccess//g' \
    -e 's/-Wno-redundant-move//g' \
    -e 's/-Wno-pessimizing-move//g' \
    -e 's/-Wno-noexcept-type//g' \
    "$BUILD_DIR/compile_commands.json" > "$TEMP_BUILD_DIR/compile_commands.json"

# Default files if none specified
if [[ ${#FILES[@]} -eq 0 ]]; then
    FILES=(
        "$PROJECT_DIR/parser/src/"*.cpp
        "$PROJECT_DIR/compiler/src/"*.cpp
        "$PROJECT_DIR/repl/src/"*.cpp
        "$PROJECT_DIR/mlir/lib/"*/*.cpp
    )
fi

# Run clang-tidy in parallel
echo "Running clang-tidy on ${#FILES[@]} files..."
printf '%s\n' "${FILES[@]}" | xargs -P"$(nproc)" -I{} \
    clang-tidy -p "$TEMP_BUILD_DIR" $FIX_FLAG {} 2>&1 | \
    grep -v "warnings generated" | \
    grep -v "Suppressed" | \
    grep -v "Use -header-filter" || true

echo "Done."
