# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Use clangd to find and fix bugs efficiently.

## Docker Environment

Project is to be built inside docker container.
You can start a container if it is not running.

```bash
# Start a container
docker/docker_run.sh
```

And you can execute commands inside the container using run_docker_command.sh.
```bash
# Run any command inside the docker container
docker/run_docker_command.sh <command> [options]
```

## clangd Commands

```bash
# Run clangd (inside docker container)
clangd --path-mappings=$(pwd)=/workspace/polang --enable-config
```

## Code Style

- Make variables and functions `const` whenever possible
- Mark functions `noexcept` when they don't throw exceptions
- Prefer `const` references for function parameters that aren't modified

## clang-format Commands

Always apply clang-format to files you edit (C/C++ files only, not .l or .y files).

```bash
# Run clang-format (inside docker container)
clang-format -i <path/to/edited-file>
```

## Build Commands

```bash
# Configure (inside docker container)
cmake -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"

# Build (inside docker container)
cmake --build build -j$(nproc)
```

Build outputs:
- `build/bin/PolangCompiler` - Compiler executable (outputs LLVM IR to stdout)
- `build/bin/PolangRepl` - Interactive REPL executable (executes code via JIT)
- `build/lib/libPolangParser.a` - Parser static library
- `build/lib/libPolangMLIRCodegen.a` - MLIR-based code generation library
- `build/lib/libPolangDialect.a` - Polang MLIR dialect library

## Dependencies

- CMake 3.20+
- LLVM 20+ (core, support, native, OrcJIT components)
- MLIR (included with LLVM 20+)
- Bison (for parser generation)
- Flex (for lexer generation)

## Documentation

When modifying the language syntax (lexer.l, parser.y, or node.hpp), always update:
- `doc/syntax.md` - Language syntax reference

When modifying the MLIR code generation pipeline, update:
- `doc/Lowering.md` - MLIR lowering process documentation

When adding new language features or modifying compiler output, consider adding lit tests:
- `tests/lit/AST/` - AST dump tests using `--dump-ast`
- `tests/lit/MLIR/` - MLIR output tests using `--emit-mlir`
- `tests/lit/LLVMIR/` - LLVM IR output tests
- `tests/lit/Execution/` - REPL execution tests
- `tests/lit/Errors/` - Error handling tests

### Lit Test Coverage (66 tests)

**AST Tests (11 files):** literals, double-literals, bool-literals, variables, mutable-variables, assignment, functions, control-flow, comparisons, let-expressions, method-calls

**MLIR Tests (16 files):** constants, arithmetic, double-arithmetic, functions, function-calls, control-flow, comparisons, types, mutable-variables, let-expressions, type-inference-literals, type-inference-expressions, type-inference-functions, type-inference-if, type-inference-params, type-inference-params-callsite

**LLVMIR Tests (13 files):** arithmetic, types, bool-type, functions, control-flow, comparisons, double-comparisons, mutable-variables, let-expressions, variable-shadowing, recursive-function, constant-folding, nested-if

**Execution Tests (12 files):** hello, variables, mutability, functions, conditionals, closures, fibonacci, factorial, double-operations, bool-operations, let-expressions, comparison-operators

**Error Tests (14 files):** syntax-errors, undefined-variable, type-mismatch, immutable-reassignment, undefined-function, function-arity, missing-else, syntax-error-paren, type-error-if-condition, return-type-mismatch, if-branch-type-mismatch, argument-type-mismatch, assignment-type-mismatch, param-type-inference-error

## Testing

After modifying the application, always verify that example programs still work:

```bash
# Run all example programs (inside docker container)
for f in example/*.po; do echo "=== $(basename $f) ==="; ./build/bin/PolangRepl "$f"; done
```

Expected outputs:
- `closures.po` → `21 : int`
- `conditionals.po` → `10 : int`
- `factorial.po` → `120 : int`
- `fibonacci.po` → `5 : int`
- `functions.po` → `25 : int`
- `hello.po` → `7 : int`
- `let_expressions.po` → `16 : int`
- `mutability.po` → `23 : int`
- `types.po` → `84 : int`
- `variables.po` → `30 : int`

Also run the test suite:

```bash
# Run all tests (inside docker container)
ctest --test-dir build --output-on-failure

# Run only lit tests (inside docker container)
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v build/tests/lit

# Run lit tests via CMake target (inside docker container)
cmake --build build --target check-polang-lit

# Run specific lit test category (inside docker container)
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v build/tests/lit/MLIR
```

### Writing Lit Tests

Lit tests use FileCheck to verify compiler output. Example test format:

```
; RUN: %polang_compiler --dump-ast %s | %FileCheck %s

; Test integer literal AST
; CHECK:      NBlock
; CHECK-NEXT: `-NExpressionStatement
; CHECK-NEXT:   `-NInteger 42
42
```

For LLVM IR tests where only part of the line needs to match:
```
; RUN: %polang_compiler %s | %FileCheck %s

; CHECK: {{.*}}alloca i64{{.*}}
; CHECK: {{.*}}store i64 1{{.*}}
let x = 1
```

Available substitutions:
- `%polang_compiler` - Path to PolangCompiler
- `%polang_repl` - Path to PolangRepl
- `%FileCheck` - Path to FileCheck
- `%not` - Inverts exit code (for error tests)
- `%s` - Current test file path

FileCheck patterns:
- `; CHECK:` - Match full line (can skip lines)
- `; CHECK-NEXT:` - Match on the immediately following line (preferred for consecutive output)
- `%{{[0-9]+}}` - Match SSA values like `%0`, `%1`
- `{{.*}}` - Match any characters (use at start/end for partial line matching)
- `{{^}}` - Match start of line (useful to disambiguate nested vs top-level patterns)
- Use `%not` for tests that should fail (e.g., syntax errors)

**Note:** FileCheck is configured with `--match-full-lines`, so patterns must match the entire line. Use `{{.*}}` at the start or end of patterns when only checking for a substring.

**Best practices:**
- Prefer `CHECK-NEXT` over `CHECK` when testing consecutive lines
- Use exact full-line patterns when possible for maximum precision
- Use `{{.*}}pattern{{.*}}` for partial matching when full line content varies

## Project Structure

```
polang/
├── CMakeLists.txt              # Root CMake configuration
├── doc/                        # Documentation
│   ├── syntax.md               # Language syntax reference
│   └── Lowering.md             # MLIR lowering process
├── mlir/                       # MLIR dialect and code generation
│   ├── CMakeLists.txt
│   ├── include/polang/
│   │   ├── Dialect/            # Polang dialect definitions (.td files)
│   │   ├── Conversion/         # Lowering pass headers
│   │   └── MLIRGen.h           # AST to MLIR interface
│   └── lib/
│       ├── Dialect/            # Dialect implementation
│       ├── Conversion/         # PolangToStandard lowering pass
│       └── MLIRGen/            # AST to MLIR visitor
├── parser/                     # Parser library
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── lexer.l             # Flex lexer
│   │   ├── parser.y            # Bison grammar
│   │   ├── parser_api.cpp      # Parser API implementation
│   │   ├── ast_printer.cpp     # AST printer implementation
│   │   ├── type_checker.cpp    # Type checking and inference
│   │   ├── operator_utils.cpp  # Operator utility functions
│   │   └── error_reporter.cpp  # Unified error reporting
│   └── include/parser/
│       ├── node.hpp            # AST node definitions
│       ├── parser_api.hpp      # Parser API header
│       ├── ast_printer.hpp     # AST printer header
│       ├── visitor.hpp         # Visitor pattern base class
│       ├── polang_types.hpp    # Type constants and utilities
│       ├── operator_utils.hpp  # Operator classification utilities
│       └── error_reporter.hpp  # Error reporting interface
├── compiler/                   # Compiler application
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp            # Compiler entry point
│   │   └── mlir_codegen.cpp    # MLIR-based code generation
│   └── include/compiler/
│       └── mlir_codegen.hpp    # MLIR code generation header
├── repl/                       # REPL application
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp            # REPL entry point
│   │   └── repl_session.cpp    # REPL session management
│   └── include/repl/
│       ├── repl_session.hpp    # REPL session header
│       └── input_checker.hpp   # Multi-line input detection
├── tests/                      # Test suite (organized by component)
│   ├── CMakeLists.txt          # Root test config (fetches GTest)
│   ├── common/
│   │   └── process_helper.hpp  # Shared test utilities
│   ├── parser/                 # Parser unit tests (11 test files)
│   │   ├── lexer_test.cpp      # Lexer tokenization tests
│   │   ├── parser_*_test.cpp   # Parser tests (declaration, expression, etc.)
│   │   ├── type_check_test.cpp # Type checker tests
│   │   ├── polang_types_test.cpp    # Type constants tests
│   │   ├── operator_utils_test.cpp  # Operator utilities tests
│   │   └── error_reporter_test.cpp  # Error reporter tests
│   ├── compiler/               # Compiler tests
│   │   └── compiler_test.cpp   # LLVM IR generation tests
│   ├── repl/                   # REPL tests
│   │   ├── repl_test.cpp       # REPL execution tests
│   │   └── repl_unit_test.cpp  # InputChecker unit tests
│   └── lit/                    # llvm-lit FileCheck tests (63 tests)
│       ├── lit.cfg.py          # Lit configuration
│       ├── lit.site.cfg.py.in  # CMake-configured site settings
│       ├── AST/                # AST dump tests (11 files)
│       ├── MLIR/               # MLIR output tests (14 files)
│       ├── LLVMIR/             # LLVM IR output tests (13 files)
│       ├── Execution/          # REPL execution tests (12 files)
│       └── Errors/             # Error handling tests (13 files)
├── example/                    # Example programs
│   ├── hello.po
│   ├── variables.po
│   ├── functions.po
│   ├── conditionals.po
│   ├── let_expressions.po
│   ├── types.po
│   ├── fibonacci.po
│   ├── factorial.po
│   ├── mutability.po
│   └── closures.po
└── docker/                     # Docker build environment
```

## Architecture

Polang is a simple programming language compiler with an MLIR-based backend.

### Components

1. **Parser Library** (`parser/`)
   - Receives source code string via `polang_parse()` API
   - Returns AST (`NBlock*`)
   - Built as static library `libPolangParser.a`
   - Includes type checking, AST printing, and error reporting utilities
   - Key modules:
     - `polang_types.hpp` - Centralized type constants (`TypeNames::INT`, etc.)
     - `operator_utils.hpp` - Operator classification and string conversion
     - `error_reporter.hpp` - Unified error reporting across all components
     - `visitor.hpp` - Visitor pattern base class for AST traversal

2. **MLIR Dialect** (`mlir/`)
   - Custom Polang dialect with types (`!polang.int`, `!polang.double`, `!polang.bool`)
   - High-level operations that mirror language semantics
   - Lowering pass to standard MLIR dialects (arith, func, scf, memref)

3. **Compiler** (`compiler/`)
   - Reads source from stdin or file
   - Uses parser library to build AST
   - MLIR backend generates Polang dialect, lowers to LLVM IR
   - Outputs IR to stdout

4. **REPL** (`repl/`)
   - Interactive read-eval-print loop
   - Links directly to parser and codegen libraries
   - Supports multi-line input for incomplete expressions
   - Persists variables and functions across evaluations
   - Executes code via LLVM JIT

### Pipeline (MLIR Backend - Default)

1. **Lexer** (`parser/src/lexer.l`) - Flex-based tokenizer
2. **Parser** (`parser/src/parser.y`) - Bison grammar that builds AST
3. **AST** (`parser/include/parser/node.hpp`) - Node class hierarchy
4. **Type Checking** (`parser/src/type_checker.cpp`) - Validates types and infers missing annotations
5. **MLIR Generation** (`mlir/lib/MLIRGen/`) - Generates Polang dialect MLIR from AST
6. **Lowering** (`mlir/lib/Conversion/`) - Lowers Polang dialect to standard dialects, then to LLVM dialect
7. **LLVM IR** - Translates LLVM dialect to LLVM IR

See `doc/Lowering.md` for detailed documentation of the MLIR lowering process.

### Key Types

- `NBlock` - Contains a `StatementList`; serves as the root AST node
- `MLIRCodeGenContext` - MLIR-based code generation context
- `Visitor` - Base class for AST visitors (type checking, code generation, etc.)
- `ErrorReporter` - Unified error reporting with severity levels and location info
- `TypeNames` - Compile-time type name constants (`INT`, `DOUBLE`, `BOOL`, etc.)

### Generated Files

Bison and Flex generate files in `build/parser/`:
- `parser.cpp`, `parser.hpp` - Parser implementation and token definitions
- `lexer.cpp` - Lexer implementation

## Usage

```bash
# Compile source to LLVM IR
echo "let x = 5" | ./build/bin/PolangCompiler

# Dump AST (for debugging)
echo "let x = 5" | ./build/bin/PolangCompiler --dump-ast

# Output Polang dialect MLIR (for debugging)
echo "let x = 5" | ./build/bin/PolangCompiler --emit-mlir

# Execute source file via REPL
./build/bin/PolangRepl example/hello.po

# Execute source via REPL (pipe mode)
echo "let x = 5" | ./build/bin/PolangRepl

# Interactive REPL session
./build/bin/PolangRepl
# > 1 + 2
# 3 : int
# > let x = 5
# > x * 2
# 10 : int
# > exit
```

### Compiler Flags

| Flag | Description |
|------|-------------|
| (default) | Output LLVM IR |
| `--dump-ast` | Dump AST and exit (no code generation) |
| `--emit-mlir` | Output Polang dialect MLIR instead of LLVM IR |
| `--help` | Show help message |
