# Testing and CI/CD

This document describes the test infrastructure, coverage analysis, and continuous integration pipeline for the Polang project.

## Test Infrastructure

### Test Frameworks

- **GoogleTest/GoogleMock**: Unit and integration tests for C++ components
- **LLVM lit**: FileCheck-based tests for compiler output verification

### Running Tests

```bash
# Run all tests (inside docker container)
ctest --test-dir build --output-on-failure

# Run only lit tests
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v build/tests/lit

# Run specific test category
ctest --test-dir build -R "CompilerIntegration"
```

## Test Categories

### Parser Tests (`tests/parser/`)

Unit tests for the lexer, parser, and type checker.

| Test File | Description |
|-----------|-------------|
| `lexer_test.cpp` | Token recognition and lexer behavior |
| `parser_declaration_test.cpp` | Variable and function declaration parsing |
| `parser_expression_test.cpp` | Expression parsing (arithmetic, comparisons) |
| `parser_control_flow_test.cpp` | If expressions and control flow |
| `parser_statement_test.cpp` | Statement parsing |
| `error_test.cpp` | Syntax error detection and reporting |
| `type_check_test.cpp` | Type checking, error detection, capture analysis |
| `ast_printer_test.cpp` | AST serialization |
| `polang_types_test.cpp` | Type constant utilities |
| `operator_utils_test.cpp` | Operator classification |
| `error_reporter_test.cpp` | Error message formatting |

### Compiler Tests (`tests/compiler/`)

Integration tests for the LLVM IR code generation and MLIR verifier unit tests.

| Test File | Description |
|-----------|-------------|
| `compiler_test.cpp` | End-to-end compilation, CLI flags, float/cast ops |
| `mlir_verifier_test.cpp` | MLIR verifier error paths (programmatic MLIR construction) |
| `conversion_pass_test.cpp` | Polang-to-Standard conversion pass tests |
| `type_inference_pass_test.cpp` | Type inference pass tests |

### REPL Tests (`tests/repl/`)

| Test File | Description |
|-----------|-------------|
| `repl_test.cpp` | REPL execution and state persistence |
| `repl_unit_test.cpp` | `InputChecker` and `EvalResult` unit tests |

### Lit Tests (`tests/lit/`)

FileCheck-based tests organized by output type:

| Directory | Count | Description |
|-----------|-------|-------------|
| `AST/` | 18 | AST dump verification (`--dump-ast`) |
| `MLIR/` | 39 | Polang dialect MLIR output (`--emit-mlir`) |
| `LLVMIR/` | 16 | LLVM IR generation |
| `Execution/` | 55 | REPL execution results |
| `Errors/` | 15 | Error message verification |

**Total: 143 lit tests**

## Writing Lit Tests

Lit tests use FileCheck to verify compiler output:

```
; RUN: %polang_compiler --dump-ast %s | %FileCheck %s

; Test integer literal AST
; CHECK:      NBlock
; CHECK-NEXT: `-NExpressionStatement
; CHECK-NEXT:   `-NInteger 42
42
```

**Error tests with source locations:**

Error messages include line and column information. Error tests verify both the message and the location:

```
; RUN: %not %polang_compiler %s 2>&1 | %FileCheck %s

; Test undefined variable error
; CHECK: Type error: Undeclared variable: x at line 5, column 1
x + 1
```

Note: The line number in the CHECK pattern must match the actual line in the test file where the error occurs.

**Available substitutions:**

| Substitution | Description |
|--------------|-------------|
| `%polang_compiler` | Path to PolangCompiler |
| `%polang_repl` | Path to PolangRepl |
| `%FileCheck` | Path to FileCheck |
| `%not` | Inverts exit code (for error tests) |
| `%s` | Current test file path |

**FileCheck patterns:**

| Pattern | Description |
|---------|-------------|
| `; CHECK:` | Match full line (can skip lines) |
| `; CHECK-NEXT:` | Match immediately following line |
| `%{{[0-9]+}}` | Match SSA values like `%0`, `%1` |
| `{{.*}}` | Match any characters |
| `{{^}}` | Match start of line |

**Note:** FileCheck is configured with `--match-full-lines`. Use `{{.*}}` for partial matching.

**Best practices:**
- Prefer `CHECK-NEXT` over `CHECK` for consecutive lines
- Use exact full-line patterns when possible
- Use `{{.*}}pattern{{.*}}` for partial matching

### Adding Lit Tests

When adding features or fixing bugs, add corresponding lit tests:

| Test Type | Directory | Flag |
|-----------|-----------|------|
| AST dump | `tests/lit/AST/` | `--dump-ast` |
| MLIR output | `tests/lit/MLIR/` | `--emit-mlir` |
| LLVM IR | `tests/lit/LLVMIR/` | (default) |
| Execution | `tests/lit/Execution/` | REPL |
| Errors | `tests/lit/Errors/` | `%not` |

### Adding a GTest

Add tests to the appropriate `tests/*/` file using existing patterns:

```cpp
TEST(Category, TestName) {
  const auto result = runCompiler("let x = 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("expected"));
}
```

## Code Coverage

### Measuring Coverage

```bash
# Configure with coverage enabled (Debug build required for accurate coverage)
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DPOLANG_ENABLE_COVERAGE=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"

# Build and run tests
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Generate HTML coverage report
cmake --build build --target coverage
```

The HTML report is generated at `build/coverage_html/index.html`.

### Current Coverage

- **Lines:** ~90% (target)
- **Functions:** ~92%

### Adding Coverage for New Code

When adding new features:

1. Add lit tests for compiler output verification
2. Add unit tests for edge cases and error handling
3. Run coverage to identify untested paths:
   ```bash
   cmake --build build --target coverage
   ```
4. Review `build/coverage_html/index.html` for uncovered lines

## Intentionally Uncovered Code

Some functions are intentionally not covered by tests because they are infrastructure code required by MLIR/LLVM but not used in our execution path.

### MLIR Pass Infrastructure (7 functions)

These methods are required by MLIR's pass infrastructure for CLI tools like `mlir-opt`, but Polang runs passes programmatically.

| File | Function | Purpose |
|------|----------|---------|
| `mlir/lib/Conversion/PolangToStandard.cpp` | `PolangToStandardPass::getArgument()` | Returns pass name for CLI |
| `mlir/lib/Conversion/PolangToStandard.cpp` | `PolangToStandardPass::getDescription()` | Returns pass description |
| `mlir/lib/Conversion/PolangToStandard.cpp` | `registerPolangConversionPasses()` | Registers passes with MLIR CLI |
| `mlir/lib/Transforms/TypeInference.cpp` | `TypeInferencePass::getArgument()` | Returns pass name for CLI |
| `mlir/lib/Transforms/TypeInference.cpp` | `TypeInferencePass::getDescription()` | Returns pass description |
| `mlir/lib/Transforms/Monomorphization.cpp` | `MonomorphizationPass::getArgument()` | Returns pass name for CLI |
| `mlir/lib/Transforms/Monomorphization.cpp` | `MonomorphizationPass::getDescription()` | Returns pass description |

### MLIR Op Interface Methods (6 functions)

Required by MLIR's `CallOpInterface` for call graph analysis, but not used since we build MLIR programmatically rather than parsing MLIR text.

| File | Function | Purpose |
|------|----------|---------|
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::getArgOperands()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::getArgOperandsMutable()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::getCallableForCallee()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::getCalleeType()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::setCalleeFromCallable()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `FuncOp::parse()` | MLIR text parsing |

### MLIR Text Parsing Methods (4 functions)

These `parse()` and `print()` methods implement MLIR textual format for ops/types. They are required by the MLIR framework but not exercised because Polang constructs MLIR programmatically (never parses MLIR text).

| File | Function | Purpose |
|------|----------|---------|
| `mlir/lib/Dialect/PolangOps.cpp` | `ConstantIntegerOp::parse()` | MLIR text parsing for integer constants |
| `mlir/lib/Dialect/PolangOps.cpp` | `ConstantFloatOp::parse()` | MLIR text parsing for float constants |
| `mlir/lib/Dialect/PolangOps.cpp` | `FuncOp::print()` | MLIR text printing for functions |
| `mlir/lib/Dialect/PolangTypes.cpp` | `TypeVarType::parse()` | MLIR text parsing for type variables |

### PrintOp Lowering (1 function)

| File | Function | Reason |
|------|----------|--------|
| `mlir/lib/Conversion/PolangToStandard.cpp` | `PrintOpLowering::matchAndRewrite()` | Print operation not in language |

### Type Checker Visitor Stubs (6 functions)

Internal visitor classes in `parser/src/type_checker.cpp` implement the visitor pattern. Some methods are intentional no-ops because the visitor only processes specific node types.

**FreeVariableCollector** (6 stubs): Identifies free variables in closures for capture analysis. Only processes `NIdentifier`, `NBlock`, `NLetExpression`, `NIfExpression`, `NBinaryOperator`, `NMethodCall`, and `NAssignment` to find free variables. The following methods are no-ops:
- `visit(NQualifiedName&)` - qualified names reference module members, not captures
- `visit(NVariableDeclaration&)` - only reached via NBlock statements
- `visit(NFunctionDeclaration&)` - nested functions have own capture analysis
- `visit(NModuleDeclaration&)` - modules don't appear in function bodies
- `visit(NImportStatement&)` - imports don't appear in function bodies
- `visit(NInteger&)`, `visit(NDouble&)`, `visit(NBoolean&)` - literals can't be free variables

### Virtual Destructors (2 functions)

| File | Function | Reason |
|------|----------|--------|
| `parser/include/parser/node.hpp` | `Node::~Node()` | Called via derived class destructors |
| `parser/include/parser/visitor.hpp` | `Visitor::~Visitor()` | Called via derived class destructors |

## Continuous Integration

GitHub Actions workflows run automatically on push and pull requests to `main`. All CI jobs run inside the project's Docker container (`ghcr.io/<owner>/polang-dev`).

### CI Pipeline (`.github/workflows/ci.yml`)

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

### Build Matrix

The `build-and-test` job runs 4 configurations:

| Compiler | Build Type |
|----------|------------|
| GCC | Debug |
| GCC | Release |
| Clang 20 | Debug |
| Clang 20 | Release |

**Build Types:**

| Type | Optimization | Debug Symbols | Assertions | Use Case |
|------|-------------|---------------|------------|----------|
| `Debug` | `-O0` | Yes (`-g`) | Enabled | Development, debugging |
| `Release` | `-O3` | No | Disabled (`NDEBUG`) | Production, benchmarking |
| `RelWithDebInfo` | `-O2` | Yes (`-g`) | Disabled | Performance profiling |

### Sanitizers

Memory and undefined behavior checking with 2 configurations:

| Sanitizer | Checks |
|-----------|--------|
| AddressSanitizer (ASan) | Memory leaks, buffer overflows, use-after-free |
| UndefinedBehaviorSanitizer (UBSan) | Integer overflow, null pointer dereference, etc. |

- Uses Clang 20 compiler (required for sanitizers)
- Debug build for better stack traces
- Configured with `halt_on_error=1` to fail fast

### Docker Image

The CI uses a Docker image (`ghcr.io/<owner>/polang-dev:latest`) containing:

- Ubuntu 24.04 base
- GCC and Clang 20 compilers
- CMake, Bison, Flex
- LLVM 20 with MLIR
- clang-format, clang-tidy, clangd
- lcov (for coverage)
- Python 3 (for lit tests)

The Docker image is rebuilt when:
- Files in `docker/**` directory are modified
- Push to `main` branch (image is pushed to GHCR)
- Pull request with docker changes (image is built but not pushed)

### Secrets

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
docker exec polang ./scripts/run-clang-format.sh --check

# Run clang-tidy
docker exec polang bash -c "cmake -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20 && ./scripts/run-clang-tidy.sh"

# Build and test (Debug, GCC - default)
docker exec polang bash -c "cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20 && cmake --build build -j\$(nproc) && ctest --test-dir build --output-on-failure"

# Build and test (Release, GCC)
docker exec polang bash -c "cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20 && cmake --build build -j\$(nproc) && ctest --test-dir build --output-on-failure"

# Build and test (Debug, Clang)
docker exec polang bash -c "cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20 && cmake --build build -j\$(nproc) && ctest --test-dir build --output-on-failure"

# Run with AddressSanitizer
docker exec polang bash -c "cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20 -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g' -DCMAKE_C_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g' -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address' && cmake --build build -j\$(nproc) && ctest --test-dir build --output-on-failure"

# Generate coverage report
docker exec polang bash -c "cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20 -DPOLANG_ENABLE_COVERAGE=ON && cmake --build build -j\$(nproc) && ctest --test-dir build --output-on-failure && cmake --build build --target coverage"
```

## Troubleshooting

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

### Docker Image Not Found

If CI fails with "image not found", the Docker image may not exist in GHCR yet:

1. Manually trigger the Docker build workflow
2. Or modify a file in `docker/` and push to `main`
