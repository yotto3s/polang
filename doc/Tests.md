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

### Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions:

- **Code quality**: clang-format and clang-tidy checks
- **Build matrix**: GCC/Clang × Debug/Release (4 configurations)
- **Sanitizers**: AddressSanitizer and UndefinedBehaviorSanitizer
- **Coverage**: Uploaded to Codecov automatically

See `doc/CI_CD.md` for detailed CI documentation.

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
| `AST/` | 19 | AST dump verification (`--dump-ast`) |
| `MLIR/` | 31 | Polang dialect MLIR output (`--emit-mlir`) |
| `LLVMIR/` | 13 | LLVM IR generation |
| `Execution/` | 28 | REPL execution results |
| `Errors/` | 13 | Error message verification |

**Total: 104 lit tests**

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

- **Lines:** ~86% (2314 of 2680 lines)
- **Functions:** ~92% (280 of 304 functions)

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
