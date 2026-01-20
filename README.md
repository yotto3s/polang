# Polang

A simple programming language with ML-inspired syntax and an MLIR/LLVM backend.

## Features

- **ML-inspired syntax**: Let bindings, type annotations, first-class functions
- **Type inference**: Hindley-Milner style type inference with polymorphism
- **MLIR backend**: Custom Polang dialect lowered through MLIR to LLVM IR
- **Interactive REPL**: JIT compilation with state persistence
- **Comprehensive testing**: Unit tests, lit tests, sanitizers, and coverage

## Quick Start

### Building

The project uses Docker for a consistent build environment:

```bash
# Start the development container
docker/docker_run.sh

# Configure and build (inside container)
cmake -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"
cmake --build build -j$(nproc)
```

Or from outside the container:

```bash
docker exec polang cmake -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"
docker exec polang cmake --build build -j$(nproc)
```

### Running

**Interactive REPL:**
```bash
./build/bin/PolangRepl
```

**Example session:**
```
$ ./build/bin/PolangRepl
Polang REPL (type 'exit' or Ctrl+D to quit)
> 1 + 2
3 : int
> let x = 5
> x * 2
10 : int
> let double (n: int): int = n * 2
> double(21)
42 : int
> if x > 3 then true else false
true : bool
> exit
```

**Compile to LLVM IR:**
```bash
echo "let x = 42" | ./build/bin/PolangCompiler
```

**Run a source file:**
```bash
./build/bin/PolangRepl example/fibonacci.po
```

## Testing

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run example programs
for f in example/*.po; do echo "=== $(basename $f) ==="; ./build/bin/PolangRepl "$f"; done
```

## Documentation

| Document | Description |
|----------|-------------|
| [Syntax](doc/Syntax.md) | Language syntax reference |
| [TypeSystem](doc/TypeSystem.md) | Type system and inference |
| [Architecture](doc/Architecture.md) | Project structure, components, MLIR lowering |
| [Building](doc/Building.md) | Build instructions, dependencies |
| [Development](doc/Development.md) | Code style and tooling |
| [Testing](doc/Testing.md) | Test infrastructure and CI/CD |

## Project Structure

```
polang/
├── parser/          # Lexer, parser, type checker
├── compiler/        # LLVM IR compiler
├── repl/            # Interactive REPL with JIT
├── mlir/            # Polang MLIR dialect and lowering
├── tests/           # Unit tests and lit tests
├── example/         # Example programs
├── doc/             # Documentation
├── scripts/         # Development scripts
└── docker/          # Docker build environment
```

## Dependencies

- CMake 3.20+
- LLVM 20+ with MLIR
- Bison and Flex
- GCC or Clang

## License

See [LICENSE](LICENSE) for details.
