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
- `build/lib/libPolangCodegen.a` - Legacy LLVM IR code generation library
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
```

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
│   │   └── parser_api.cpp      # Parser API implementation
│   └── include/parser/
│       ├── node.hpp            # AST node definitions
│       └── parser_api.hpp      # Parser API header
├── compiler/                   # Compiler application
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp            # Compiler entry point
│   │   ├── codegen.cpp         # Legacy LLVM IR code generation
│   │   └── mlir_codegen.cpp    # MLIR-based code generation
│   └── include/compiler/
│       ├── codegen.hpp         # Legacy code generation header
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
│   ├── parser/                 # Parser unit tests
│   │   └── *_test.cpp          # Lexer, parser, type checker tests
│   ├── compiler/               # Compiler tests
│   │   └── compiler_test.cpp   # LLVM IR generation tests
│   └── repl/                   # REPL tests
│       ├── repl_test.cpp       # REPL execution tests
│       └── repl_unit_test.cpp  # InputChecker unit tests
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

2. **MLIR Dialect** (`mlir/`)
   - Custom Polang dialect with types (`!polang.int`, `!polang.double`, `!polang.bool`)
   - High-level operations that mirror language semantics
   - Lowering pass to standard MLIR dialects (arith, func, scf, memref)

3. **Compiler** (`compiler/`)
   - Reads source from stdin or file
   - Uses parser library to build AST
   - Default: MLIR backend generates Polang dialect, lowers to LLVM IR
   - Legacy: Direct LLVM IR generation (with `--legacy` flag)
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
4. **MLIR Generation** (`mlir/lib/MLIRGen/`) - Generates Polang dialect MLIR from AST
5. **Lowering** (`mlir/lib/Conversion/`) - Lowers Polang dialect to standard dialects, then to LLVM dialect
6. **LLVM IR** - Translates LLVM dialect to LLVM IR

See `doc/Lowering.md` for detailed documentation of the MLIR lowering process.

### Key Types

- `NBlock` - Contains a `StatementList`; serves as the root AST node
- `MLIRCodeGenContext` - MLIR-based code generation context (default backend)
- `CodeGenContext` - Legacy LLVM IR code generation context (with `--legacy`)

### Generated Files

Bison and Flex generate files in `build/parser/`:
- `parser.cpp`, `parser.hpp` - Parser implementation and token definitions
- `lexer.cpp` - Lexer implementation

## Usage

```bash
# Compile source to LLVM IR (using MLIR backend)
echo "let x = 5" | ./build/bin/PolangCompiler

# Compile with legacy LLVM IR backend
echo "let x = 5" | ./build/bin/PolangCompiler --legacy

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
| (default) | Use MLIR backend, output LLVM IR |
| `--emit-mlir` | Output Polang dialect MLIR instead of LLVM IR |
| `--legacy` | Use legacy direct LLVM IR backend |
| `--help` | Show help message |
