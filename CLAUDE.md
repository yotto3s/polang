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

## clang-format Commands

Always apply clang-format to files you edit.

```bash
# Run clang-format (inside docker container)
clang-format -i <path/to/edited-file>
```

## Build Commands

```bash
# Configure (inside docker container)
cmake -S. -Bbuild

# Build (inside docker container)
cmake --build build
```

The executable is built at `build/src/PolangRepl`.

## Dependencies

- CMake 3.10+
- LLVM (core, support, native components)
- Bison (for parser generation)
- Flex (for lexer generation)

## Architecture

Polang is a simple programming language compiler with LLVM backend.

### Pipeline

1. **Lexer** (`src/lexer.l`) - Flex-based tokenizer recognizing identifiers, integers, doubles, operators, and punctuation
2. **Parser** (`src/parser.y`) - Bison grammar that builds an AST from tokens; outputs `programBlock` (NBlock*)
3. **AST** (`src/node.hpp`) - Node class hierarchy: `NExpression`, `NStatement`, and concrete types like `NInteger`, `NIdentifier`, `NBinaryOperator`, `NFunctionDeclaration`, etc.
4. **Code Generation** (`src/codegen.cpp`, `src/codegen.hpp`) - `CodeGenContext` traverses AST and emits LLVM IR via `codeGen()` virtual methods on each node type

### Key Types

- `NBlock` - Contains a `StatementList`; serves as the root AST node
- `CodeGenContext` - Manages LLVM module, block stack, and local variable scopes
- `CodeGenBlock` - Holds a `BasicBlock*` and local variable map for scope tracking

### Generated Files

Bison and Flex generate files in `build/src/`:
- `parser.cpp`, `parser.hpp` - Parser implementation and token definitions
- `lexer.cpp` - Lexer implementation
