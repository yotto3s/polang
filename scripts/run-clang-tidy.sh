#!/bin/bash
# Run clang-tidy on the Polang codebase
# Usage: ./scripts/run-clang-tidy.sh [--fix] [files...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

# Check if build directory exists
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: Build directory not found. Run cmake first."
    exit 1
fi

# Check if compile_commands.json exists
if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: compile_commands.json not found. Configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    exit 1
fi

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
            FILES+=("$1")
            shift
            ;;
    esac
done

# Create filtered compile_commands.json (remove GCC-specific flags)
FILTERED_COMPILE_COMMANDS=$(mktemp)
trap "rm -f $FILTERED_COMPILE_COMMANDS" EXIT

sed -e 's/-fno-lifetime-dse//g' \
    -e 's/-Wno-class-memaccess//g' \
    -e 's/-Wno-redundant-move//g' \
    -e 's/-Wno-pessimizing-move//g' \
    -e 's/-Wno-noexcept-type//g' \
    "$BUILD_DIR/compile_commands.json" > "$FILTERED_COMPILE_COMMANDS"

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
    clang-tidy -p "$(dirname "$FILTERED_COMPILE_COMMANDS")" $FIX_FLAG {} 2>&1 | \
    grep -v "warnings generated" | \
    grep -v "Suppressed" | \
    grep -v "Use -header-filter" || true

echo "Done."
