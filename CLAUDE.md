# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

For detailed documentation, see:
- `doc/Building.md` - Build instructions, dependencies, build types
- `doc/Architecture.md` - Project structure, components, MLIR lowering pipeline
- `doc/Development.md` - Code style, tooling, testing workflow
- `doc/Syntax.md` - Language syntax reference
- `doc/TypeSystem.md` - Type system and inference
- `doc/Testing.md` - Test infrastructure, coverage, and CI/CD

## Essential Commands

```bash
# Build using presets (recommended)
cmake --preset clang-debug
cmake --build --preset clang-debug
ctest --preset clang-debug

# Format code (required before committing)
./scripts/run-clang-format.sh

# Run static analysis
./scripts/run-clang-tidy.sh

# MLIR round-trip testing tool (used by lit tests)
./build/bin/polang-opt input.mlir

# Verify examples work
for f in example/*.po; do echo "=== $(basename $f) ==="; ./build/bin/PolangRepl "$f"; done
```

Available presets: `default`, `gcc-debug`, `gcc-release`, `clang-debug`, `clang-release`, `asan`, `ubsan`, `coverage`, `lint`

## Documentation Updates

When modifying code, update the relevant documentation:

| Change Type | Documentation |
|-------------|---------------|
| Language syntax (lexer.l, parser.y, node.hpp) | `doc/Syntax.md` |
| MLIR pipeline | `doc/Architecture.md` |
| Type system | `doc/TypeSystem.md` |
| Tests / CI | `doc/Testing.md` |
| Build system | `doc/Building.md` |
| Architecture | `doc/Architecture.md` |

## MLIR Development

When modifying code under `mlir/`, in PLAN mode, **read** the official MLIR documentation to make informed decisions:

- **Main site**: https://mlir.llvm.org/
- **Deprecation notices**: https://mlir.llvm.org/deprecation/ - Check for deprecated APIs before using them
- **Documentation**: https://mlir.llvm.org/docs/ - Read dialect sections to choose the most appropriate dialect and follow recommended patterns
