# Continuous Integration

This document describes the CI/CD pipeline for the Polang project.

## Overview

GitHub Actions workflows run automatically on push and pull requests to `main`. All CI jobs run inside the project's Docker container (`ghcr.io/<owner>/polang-dev`).

## Workflows

### CI Pipeline (`.github/workflows/ci.yml`)

The main CI workflow with the following jobs:

```
                              ┌─────────────────────────────────────────┐
check-changes ─→ build-image ─┤ (only runs if docker/** changed)       │
                              │ (skipped otherwise, downstream continues)│
                              └──────────────────┬──────────────────────┘
                                                 │
format-check ────────────────────────────────────┼─→ build-and-test ─┬─→ sanitizers
                                                 │                   │
lint ────────────────────────────────────────────┘                   └─→ coverage
```

### Job Descriptions

| Job | Runs | Description |
|-----|------|-------------|
| `check-changes` | Always | Detects if `docker/**` files were modified using `dorny/paths-filter` |
| `build-image` | If docker/** changed | Builds and pushes Docker image to GHCR |
| `format-check` | Always | Verifies clang-format compliance via `scripts/run-clang-format.sh --check` |
| `lint` | Always | Runs clang-tidy static analysis via `scripts/run-clang-tidy.sh` |
| `build-and-test` | After format-check, lint, build-image | Builds and tests with GCC/Clang × Debug/Release matrix (4 parallel jobs) |
| `sanitizers` | After build-and-test | Runs tests with AddressSanitizer and UndefinedBehaviorSanitizer (2 parallel jobs) |
| `coverage` | After build-and-test | Generates code coverage report and uploads to Codecov |

### Job Details

#### check-changes

Detects changes in `docker/**` directory to determine if Docker image rebuild is needed.

- Uses `dorny/paths-filter@v3` action
- Outputs `docker-changed` boolean for downstream jobs

#### build-image

Builds and pushes the development Docker image.

- **Trigger**: Only when `docker/**` files are modified
- **Push**: Only on `main` branch (not on PRs)
- **Caching**: Uses registry cache from previous builds
- **Image tags**: `latest` (on main) and commit SHA

#### format-check

Verifies all source files comply with clang-format rules.

- Runs `./scripts/run-clang-format.sh --check`
- Checks C++ files in `parser/`, `compiler/`, `repl/`, and `mlir/`
- Does not modify files, only reports violations

#### lint

Runs clang-tidy static analysis on the codebase.

- Runs `./scripts/run-clang-tidy.sh`
- Requires CMake configuration for `compile_commands.json`
- Checks are defined in `.clang-tidy` configuration file

#### build-and-test

Main build and test matrix with 4 configurations:

| Compiler | Build Type |
|----------|------------|
| GCC | Debug |
| GCC | Release |
| Clang 20 | Debug |
| Clang 20 | Release |

Each configuration:
1. Configures CMake with appropriate compiler and build type
2. Builds the project
3. Runs unit tests via `ctest`
4. Runs all example programs

#### sanitizers

Memory and undefined behavior checking with 2 configurations:

| Sanitizer | Checks |
|-----------|--------|
| AddressSanitizer (ASan) | Memory leaks, buffer overflows, use-after-free |
| UndefinedBehaviorSanitizer (UBSan) | Integer overflow, null pointer dereference, etc. |

- Uses Clang 20 compiler (required for sanitizers)
- Debug build for better stack traces
- Configured with `halt_on_error=1` to fail fast

#### coverage

Generates and uploads code coverage report.

- Uses GCC with gcov for coverage instrumentation
- Generates `coverage.info` in lcov format
- Uploads to Codecov via `codecov/codecov-action@v4`
- Requires `CODECOV_TOKEN` secret for upload

## Docker Image

### Image Location

```
ghcr.io/<owner>/polang-dev:latest
```

### When Image is Built

The Docker image is rebuilt when:
- Files in `docker/**` directory are modified
- Push to `main` branch (image is pushed to GHCR)
- Pull request with docker changes (image is built but not pushed)

### Image Contents

- Ubuntu 24.04 base
- GCC and Clang 20 compilers
- CMake, Bison, Flex
- LLVM 20 with MLIR
- clang-format, clang-tidy, clangd
- lcov (for coverage)
- Python 3 (for lit tests)

## Secrets

The following secrets are used by CI:

| Secret | Purpose |
|--------|---------|
| `GITHUB_TOKEN` | Automatically provided, used for GHCR authentication |
| `CODECOV_TOKEN` | Required for uploading coverage reports to Codecov |

## Running CI Locally

To replicate CI checks locally using Docker:

```bash
# Start the development container
docker/docker_run.sh

# Run format check
docker/run_docker_command.sh ./scripts/run-clang-format.sh --check

# Run clang-tidy
docker/run_docker_command.sh bash -c "cmake -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20 && ./scripts/run-clang-tidy.sh"

# Build and test (Debug, GCC)
docker/run_docker_command.sh bash -c "cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j\$(nproc) && ctest --test-dir build --output-on-failure"

# Run with AddressSanitizer
docker/run_docker_command.sh bash -c "cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g' -DCMAKE_C_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g' -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address' && cmake --build build -j\$(nproc) && ctest --test-dir build --output-on-failure"

# Generate coverage report
docker/run_docker_command.sh bash -c "cmake -S. -Bbuild -DPOLANG_ENABLE_COVERAGE=ON && cmake --build build -j\$(nproc) && ctest --test-dir build --output-on-failure && cmake --build build --target coverage"
```

## Troubleshooting

### Docker Image Not Found

If CI fails with "image not found", the Docker image may not exist in GHCR yet:

1. Manually trigger the Docker build workflow
2. Or modify a file in `docker/` and push to `main`

### Format Check Failures

Run locally to see which files need formatting:

```bash
./scripts/run-clang-format.sh --check
```

Fix by running without `--check`:

```bash
./scripts/run-clang-format.sh
```

### Clang-Tidy Failures

Run locally to see detailed warnings:

```bash
./scripts/run-clang-tidy.sh
```

Fix with auto-fix (use with caution):

```bash
./scripts/run-clang-tidy.sh --fix
```

### Sanitizer Failures

Sanitizer errors include stack traces. Common issues:
- **ASan**: Memory leaks, buffer overflows
- **UBSan**: Integer overflow, null dereference

Run locally with sanitizers to reproduce and debug.
