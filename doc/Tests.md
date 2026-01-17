# Test Documentation

This document describes the test infrastructure, coverage analysis, and intentionally uncovered code in the Polang project.

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
| `type_check_test.cpp` | Type inference and checking |
| `ast_printer_test.cpp` | AST serialization |
| `polang_types_test.cpp` | Type constant utilities |
| `operator_utils_test.cpp` | Operator classification |
| `error_reporter_test.cpp` | Error message formatting |

### Compiler Tests (`tests/compiler/`)

Integration tests for the LLVM IR code generation.

| Test | Description |
|------|-------------|
| `CompilerIntegration.*` | End-to-end compilation tests |
| `CompilerCLI.HelpFlag` | CLI `--help` flag test |

### REPL Tests (`tests/repl/`)

| Test File | Description |
|-----------|-------------|
| `repl_test.cpp` | REPL execution and state persistence |
| `repl_unit_test.cpp` | `InputChecker` and `EvalResult` unit tests |

### Lit Tests (`tests/lit/`)

FileCheck-based tests organized by output type:

| Directory | Count | Description |
|-----------|-------|-------------|
| `AST/` | 11 | AST dump verification (`--dump-ast`) |
| `MLIR/` | 30 | Polang dialect MLIR output (`--emit-mlir`) |
| `LLVMIR/` | 13 | LLVM IR generation |
| `Execution/` | 23 | REPL execution results |
| `Errors/` | 13 | Error message verification |

**Total: 90 lit tests**

## Code Coverage

### Measuring Coverage

```bash
# Configure with coverage enabled
cmake -S. -Bbuild -DPOLANG_ENABLE_COVERAGE=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"

# Build and run tests
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Generate HTML coverage report
cmake --build build --target coverage
```

The HTML report is generated at `build/coverage_html/index.html`.

### Current Coverage

- **Lines:** ~83% (1899 of 2278 lines)
- **Functions:** ~86% (237 of 275 functions)

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

### MLIR Op Interface Methods (8 functions)

Required by MLIR's `CallOpInterface` for call graph analysis, but not used since we build MLIR programmatically rather than parsing MLIR text.

| File | Function | Purpose |
|------|----------|---------|
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::getArgOperands()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::getArgOperandsMutable()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::getCallableForCallee()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::getCalleeType()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `CallOp::setCalleeFromCallable()` | Call graph analysis |
| `mlir/lib/Dialect/PolangOps.cpp` | `FuncOp::parse()` | MLIR text parsing |
| `mlir/lib/Dialect/PolangOps.cpp` | `IfOp::parse()` | MLIR text parsing |

### PrintOp Lowering (1 function)

| File | Function | Reason |
|------|----------|--------|
| `mlir/lib/Conversion/PolangToStandard.cpp` | `PrintOpLowering::matchAndRewrite()` | Print operation not in language |

### Type Checker Visitor Stubs (21 functions)

Internal visitor classes in `parser/src/type_checker.cpp` implement the visitor pattern. Many methods are intentional no-ops because each visitor only processes specific node types.

**CallSiteCollector** (11 stubs): Only processes `NMethodCall` nodes to collect function call sites. Other visit methods return early.

**FreeVariableCollector** (4 stubs): Only processes `NIdentifier`, `NBlock`, and expression nodes to find free variables in closures.

**ParameterTypeInferrer** (5 stubs): Only processes `NInteger`, `NDouble`, `NIdentifier`, `NMethodCall`, `NBinaryOperator`, and `NIfExpression` to infer parameter types from usage.

### Virtual Destructors (2 functions)

| File | Function | Reason |
|------|----------|--------|
| `parser/include/parser/node.hpp` | `Node::~Node()` | Called via derived class destructors |
| `parser/include/parser/visitor.hpp` | `Visitor::~Visitor()` | Called via derived class destructors |

## Adding New Tests

### Adding a Lit Test

1. Create a `.po` file in the appropriate `tests/lit/` subdirectory
2. Add RUN and CHECK directives:

```
; RUN: %polang_compiler --dump-ast %s | %FileCheck %s

; CHECK: NBlock
42
```

### Adding a GTest

1. Add test to the appropriate `tests/*/` file
2. Use existing patterns:

```cpp
TEST(Category, TestName) {
  const auto result = runCompiler("let x = 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("expected"));
}
```

### Adding Coverage for New Code

When adding new features:

1. Add lit tests for compiler output verification
2. Add unit tests for edge cases and error handling
3. Run coverage to identify untested paths:
   ```bash
   cmake --build build --target coverage
   ```
4. Review `build/coverage_html/index.html` for uncovered lines
