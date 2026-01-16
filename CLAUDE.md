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
cmake -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build (inside docker container)
cmake --build build -j$(nproc)
```

Build outputs:
- `build/bin/PolangCompiler` - Compiler executable (outputs LLVM IR to stdout)
- `build/bin/PolangRepl` - Interactive REPL executable (executes code via JIT)
- `build/lib/libPolangParser.a` - Parser static library
- `build/lib/libPolangCodegen.a` - Code generation static library

## Dependencies

- CMake 3.10+
- LLVM (core, support, native, OrcJIT, AsmParser components)
- Bison (for parser generation)
- Flex (for lexer generation)

## Documentation

When modifying the language syntax (lexer.l, parser.y, or node.hpp), always update:
- `doc/syntax.md` - Language syntax reference

## Project Structure

```
polang/
├── CMakeLists.txt              # Root CMake configuration
├── doc/                        # Documentation
│   └── syntax.md               # Language syntax reference
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
│   │   └── codegen.cpp         # LLVM IR code generation
│   └── include/compiler/
│       └── codegen.hpp         # Code generation header
├── repl/                       # REPL application
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp            # REPL entry point
│   │   └── repl_session.cpp    # REPL session management
│   └── include/repl/
│       └── repl_session.hpp    # REPL session header
└── docker/                     # Docker build environment
```

## Architecture

Polang is a simple programming language compiler with LLVM backend.

### Components

1. **Parser Library** (`parser/`)
   - Receives source code string via `polang_parse()` API
   - Returns AST (`NBlock*`)
   - Built as static library `libPolangParser.a`

2. **Compiler** (`compiler/`)
   - Reads source from stdin
   - Uses parser library to build AST
   - Generates LLVM IR using `CodeGenContext`
   - Outputs IR to stdout

3. **Codegen Library** (`compiler/`)
   - `libPolangCodegen.a` - Code generation as static library
   - Used by both compiler and REPL

4. **REPL** (`repl/`)
   - Interactive read-eval-print loop
   - Links directly to parser and codegen libraries
   - Supports multi-line input for incomplete expressions
   - Persists variables and functions across evaluations
   - Executes code via LLVM JIT

### Pipeline

1. **Lexer** (`parser/src/lexer.l`) - Flex-based tokenizer recognizing identifiers, integers, doubles, operators, and punctuation
2. **Parser** (`parser/src/parser.y`) - Bison grammar that builds an AST from tokens; outputs `programBlock` (NBlock*)
3. **AST** (`parser/include/parser/node.hpp`) - Node class hierarchy: `NExpression`, `NStatement`, and concrete types like `NInteger`, `NIdentifier`, `NBinaryOperator`, `NFunctionDeclaration`, etc.
4. **Code Generation** (`compiler/src/codegen.cpp`) - `CodeGenContext` traverses AST and emits LLVM IR via `codeGen()` virtual methods on each node type

### Key Types

- `NBlock` - Contains a `StatementList`; serves as the root AST node
- `CodeGenContext` - Manages LLVM module, block stack, and local variable scopes
- `CodeGenBlock` - Holds a `BasicBlock*` and local variable map for scope tracking

### Generated Files

Bison and Flex generate files in `build/parser/`:
- `parser.cpp`, `parser.hpp` - Parser implementation and token definitions
- `lexer.cpp` - Lexer implementation

## Usage

```bash
# Compile source to LLVM IR
echo "let x = 5" | ./build/bin/PolangCompiler

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
