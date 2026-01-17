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

## Docker Environment

All commands should be run inside the Docker container:

```bash
# Start a container
docker/docker_run.sh

# Run any command inside the docker container
docker/run_docker_command.sh <command> [options]
```

## Essential Commands

```bash
# Build (Debug)
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"
cmake --build build -j$(nproc)

# Format code (required before committing)
./scripts/run-clang-format.sh

# Run static analysis
./scripts/run-clang-tidy.sh

# Run tests
ctest --test-dir build --output-on-failure

# Verify examples work
for f in example/*.po; do echo "=== $(basename $f) ==="; ./build/bin/PolangRepl "$f"; done
```

## Code Style Summary

- Use `const` whenever possible
- Mark functions `noexcept` when they don't throw
- Add `[[nodiscard]]` to functions whose return value matters
- Use braces around all control flow bodies
- Follow LLVM naming: `CamelCase` for types, `lowerCamelCase` for functions/variables

See `doc/Development.md` for full style guide.

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

## Lit Test Categories

| Directory | Count | Description |
|-----------|-------|-------------|
| `tests/lit/AST/` | 19 | AST dump tests (`--dump-ast`) |
| `tests/lit/MLIR/` | 31 | MLIR output tests (`--emit-mlir`) |
| `tests/lit/LLVMIR/` | 13 | LLVM IR generation |
| `tests/lit/Execution/` | 28 | REPL execution |
| `tests/lit/Errors/` | 13 | Error handling |

## Expected Example Outputs

| Example | Output |
|---------|--------|
| `closures.po` | `21 : int` |
| `conditionals.po` | `10 : int` |
| `factorial.po` | `120 : int` |
| `fibonacci.po` | `5 : int` |
| `functions.po` | `25 : int` |
| `hello.po` | `7 : int` |
| `let_expressions.po` | `16 : int` |
| `mutability.po` | `23 : int` |
| `types.po` | `84 : int` |
| `variables.po` | `30 : int` |

## clangd

Use clangd to find and fix bugs efficiently:

```bash
clangd --path-mappings=$(pwd)=/workspace/polang --enable-config
```
