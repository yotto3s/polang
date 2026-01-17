#!/bin/bash
# Run clang-format on the Polang codebase
# Usage: ./scripts/run-clang-format.sh [--check] [files...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
CHECK_FLAG=""
FILES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_FLAG="--dry-run --Werror"
            shift
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Default files if none specified (C++ files only, not .l or .y)
if [[ ${#FILES[@]} -eq 0 ]]; then
    FILES=(
        "$PROJECT_DIR/parser/include/parser/"*.hpp
        "$PROJECT_DIR/parser/src/"*.cpp
        "$PROJECT_DIR/compiler/include/compiler/"*.hpp
        "$PROJECT_DIR/compiler/src/"*.cpp
        "$PROJECT_DIR/repl/include/repl/"*.hpp
        "$PROJECT_DIR/repl/src/"*.cpp
        "$PROJECT_DIR/mlir/include/polang/"*.h
        "$PROJECT_DIR/mlir/lib/"*/*.cpp
    )
fi

# Run clang-format
if [[ -n "$CHECK_FLAG" ]]; then
    echo "Checking formatting on ${#FILES[@]} files..."
    clang-format $CHECK_FLAG "${FILES[@]}"
    echo "All files are properly formatted."
else
    echo "Formatting ${#FILES[@]} files..."
    clang-format -i "${FILES[@]}"
    echo "Done."
fi
