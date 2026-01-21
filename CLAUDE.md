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

## Development Workflow

When planning changes that modify the codebase, follow this test-first workflow:

1. **Create/modify tests first** - Write tests that define the expected behavior
2. **Commit the tests** - Create a commit with just the test changes
3. **Wait for review** - Pause for the user to review and approve the tests
4. **Then implement** - Start modifying the codebase to make the tests pass

This ensures agreement on expected behavior before implementation begins.

### Plan Mode Behavior

When in PLAN mode, keep asking clarifying questions until there are no unclear points in the plan. Do not exit PLAN mode until:

1. All ambiguous requirements have been clarified
2. Assumptions have been verified with the user
3. Architectural decisions and trade-offs are confirmed
4. The plan is complete and unambiguous

### After Exiting Plan Mode

Once the plan is complete and approved, immediately:

1. **Create a GitHub issue** - Include the full plan as a TODO checklist in the body
2. **Checkout a new branch** - Named appropriately for the feature/fix
3. **Create a draft PR with an empty commit** - Link it to the issue

```bash
# Create GitHub issue with full plan as TODO list
gh issue create --title "Issue title" --body "## Plan

- [ ] Step 1: Description
- [ ] Step 2: Description
- [ ] Step 3: Description
..."

# Create branch and empty commit
git checkout -b feature/issue-name
git commit --allow-empty -m "Initial commit for: issue title"

# Push and create draft PR linked to the issue
git push -u origin feature/issue-name
gh pr create --title "Issue title" --body "Closes #<issue-number>" --draft
```

This establishes the PR early for visibility and tracking before implementation begins.

## Docker Environment

All commands should be run inside the Docker container:

```bash
# Start a container
docker/docker_run.sh

# Run any command inside the docker container
docker exec polang <command> [options]
```

## Essential Commands

```bash
# Build using presets (recommended)
cmake --preset clang-debug
cmake --build --preset clang-debug
ctest --preset clang-debug

# Or build manually
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

Available presets: `default`, `gcc-debug`, `gcc-release`, `clang-debug`, `clang-release`, `asan`, `ubsan`, `coverage`, `lint`

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

## MLIR Development

When modifying code under `mlir/`, in PLAN mode, **read** the official MLIR documentation to make informed decisions:

- **Main site**: https://mlir.llvm.org/
- **Deprecation notices**: https://mlir.llvm.org/deprecation/ - Check for deprecated APIs before using them
- **Documentation**: https://mlir.llvm.org/docs/ - Read dialect sections to choose the most appropriate dialect and follow recommended patterns

## Lit Test Categories

| Directory | Count | Description |
|-----------|-------|-------------|
| `tests/lit/AST/` | 20 | AST dump tests (`--dump-ast`) |
| `tests/lit/MLIR/` | 38 | MLIR output tests (`--emit-mlir`) |
| `tests/lit/LLVMIR/` | 13 | LLVM IR generation |
| `tests/lit/Execution/` | 39 | REPL execution |
| `tests/lit/Errors/` | 16 | Error handling |

## Expected Example Outputs

| Example | Output |
|---------|--------|
| `closures.po` | `21 : i64` |
| `conditionals.po` | `10 : i64` |
| `factorial.po` | `120 : i64` |
| `fibonacci.po` | `5 : i64` |
| `functions.po` | `25 : i64` |
| `hello.po` | `7 : i64` |
| `let_expressions.po` | `16 : i64` |
| `types.po` | `84 : i64` |
| `variables.po` | `30 : i64` |

## clangd

Use clangd inside Docker for quick error checking without a full build.

### Path Mapping

Host and container paths differ:
- **Host**: `/home/yotto/dev/polang/...`
- **Container**: `/workspace/polang/...`

When using clangd, convert host paths to container paths.

### Quick Error Check

Check a single file for compilation errors (faster than full build):

```bash
# Check a specific file
docker exec polang bash -c "cd /workspace/polang && clangd --compile-commands-dir=build/clang-debug --check=/workspace/polang/mlir/lib/Conversion/PolangToStandard.cpp 2>&1 | grep -E 'error|warning|Line [0-9]+:'"

# Check with full output
docker exec polang bash -c "cd /workspace/polang && clangd --compile-commands-dir=build/clang-debug --check=/workspace/polang/<path-to-file.cpp> 2>&1"
```

### LSP Server Mode

For IDE integration with path mappings:

```bash
clangd --path-mappings=$(pwd)=/workspace/polang --enable-config
```

### Common Usage

```bash
# Check parser files
docker exec polang bash -c "cd /workspace/polang && clangd --compile-commands-dir=build/clang-debug --check=/workspace/polang/parser/src/type_checker.cpp 2>&1 | tail -10"

# Check MLIR dialect files
docker exec polang bash -c "cd /workspace/polang && clangd --compile-commands-dir=build/clang-debug --check=/workspace/polang/mlir/lib/Dialect/PolangOps.cpp 2>&1 | tail -10"

# Check conversion passes
docker exec polang bash -c "cd /workspace/polang && clangd --compile-commands-dir=build/clang-debug --check=/workspace/polang/mlir/lib/Conversion/PolangToStandard.cpp 2>&1 | tail -10"
```

Note: Requires `build/clang-debug/compile_commands.json` to exist (generated by cmake).
