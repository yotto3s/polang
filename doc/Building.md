# Building Polang

This document describes how to build the Polang compiler and its components.

## Prerequisites

### Dependencies

- CMake 3.20+
- LLVM 20+ (core, support, native, OrcJIT components)
- MLIR (included with LLVM 20+)
- Bison (for parser generation)
- Flex (for lexer generation)
- GCC or Clang compiler with C++17 support

### Docker Environment (Recommended)

The project includes a Docker environment with all dependencies pre-installed:

- Ubuntu 24.04 base
- GCC and Clang 20 compilers
- CMake, Bison, Flex
- LLVM 20 with MLIR
- clang-format, clang-tidy, clangd
- lcov (for coverage), Python 3 (for lit tests)

```bash
# Start a container
docker/docker_run.sh

# Run any command inside the docker container
docker exec polang <command> [options]

# Build the Docker image locally
docker/docker_build.sh
```

## Build Commands

All commands should be run inside the Docker container (or with equivalent dependencies installed).

### Using CMake Presets (Recommended)

The project includes CMake presets that match CI configurations:

```bash
# List available presets
cmake --list-presets

# Configure, build, and test with a preset
cmake --preset clang-debug
cmake --build --preset clang-debug
ctest --preset clang-debug
```

### Available Presets

| Preset | Compiler | Build Type | Description |
|--------|----------|------------|-------------|
| `default` | System default | Debug | Quick local development |
| `gcc-debug` | GCC | Debug | GCC Debug build |
| `gcc-release` | GCC | Release | GCC Release build |
| `clang-debug` | Clang-20 | Debug | Clang Debug build |
| `clang-release` | Clang-20 | Release | Clang Release build |
| `asan` | Clang-20 | Debug | AddressSanitizer enabled |
| `ubsan` | Clang-20 | Debug | UndefinedBehaviorSanitizer enabled |
| `coverage` | GCC | Debug | Code coverage enabled |
| `lint` | System default | Debug | For clang-tidy (generates compile_commands.json) |

Each preset creates a separate build directory: `build/<preset-name>/`

### Manual Configuration

```bash
# Configure Debug build (default)
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"

# Configure Release build
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"

# Build
cmake --build build -j$(nproc)
```

## Build Types

| Type | Optimization | Debug Symbols | Assertions | Use Case |
|------|-------------|---------------|------------|----------|
| `Debug` | `-O0` | Yes (`-g`) | Enabled | Development, debugging |
| `Release` | `-O3` | No | Disabled (`NDEBUG`) | Production, benchmarking |
| `RelWithDebInfo` | `-O2` | Yes (`-g`) | Disabled | Performance profiling |

## Build Outputs

After a successful build, the following artifacts are created:

| Output | Description |
|--------|-------------|
| `build/bin/PolangCompiler` | Compiler executable (outputs LLVM IR to stdout) |
| `build/bin/PolangRepl` | Interactive REPL executable (executes code via JIT) |
| `build/lib/libPolangParser.a` | Parser static library |
| `build/lib/libPolangMLIRCodegen.a` | MLIR-based code generation library |
| `build/lib/libPolangDialect.a` | Polang MLIR dialect library |

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | Build type (Debug, Release, RelWithDebInfo) |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | `OFF` | Generate `compile_commands.json` for tooling |
| `CMAKE_PREFIX_PATH` | - | Path to LLVM/MLIR installation |
| `POLANG_ENABLE_COVERAGE` | `OFF` | Enable code coverage with gcov |

## Code Coverage

To measure test coverage:

```bash
# Configure with coverage enabled (Debug build required)
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DPOLANG_ENABLE_COVERAGE=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"

# Build
cmake --build build -j$(nproc)

# Run tests to generate coverage data
ctest --test-dir build --output-on-failure

# Generate HTML coverage report
cmake --build build --target coverage

# Reset coverage counters
cmake --build build --target coverage-clean
```

The HTML report is generated at `build/coverage_html/index.html`.

## Generated Files

Bison and Flex generate files in `build/parser/`:

| File | Description |
|------|-------------|
| `parser.cpp` | Parser implementation |
| `parser.hpp` | Token definitions and parser interface |
| `lexer.cpp` | Lexer implementation |

## Troubleshooting

### LLVM/MLIR Not Found

Ensure `CMAKE_PREFIX_PATH` points to your LLVM installation:

```bash
cmake -S. -Bbuild -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"
```

### Shared Library Errors at Runtime

The Docker environment has LLVM library paths configured via ldconfig, so shared library errors should not occur inside the container.

If you're building outside Docker and see errors about missing `.so` files, set `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:$LD_LIBRARY_PATH
```

### Build Fails with Warnings

The project uses `-Werror` to treat warnings as errors. If you encounter build failures:

1. Check if it's a third-party header issue (may need pragma suppressions)
2. Run clang-tidy to identify and fix code issues
3. Ensure you're using a supported compiler version (GCC 11+ or Clang 20+)
