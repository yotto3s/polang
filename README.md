# Polang

A simple programming language with ML-inspired syntax and LLVM backend.

## How to Build

```bash
# Configure
cmake -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"

# Build
cmake --build build -j$(nproc)
```

## Usage

### REPL (Interactive Mode)

The REPL provides an interactive environment for evaluating Polang expressions:

```bash
./build/bin/PolangRepl
```

**Features:**
- **Interactive evaluation**: Enter expressions and see results immediately
- **State persistence**: Variables and functions persist across evaluations
- **Multi-line input**: Incomplete expressions (unbalanced parentheses, `if` without `else`, `let` without `in`) automatically continue to the next line
- **Type display**: Results are displayed with their types (e.g., `42 : int`)

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
> let y = 1
... in y + 1
2 : int
> exit
```

**Pipe mode:**
```bash
echo "1 + 2" | ./build/bin/PolangRepl
# Output: 3 : int
```

### Compiler

The compiler reads Polang source code and outputs LLVM IR:

```bash
echo "let x = 42" | ./build/bin/PolangCompiler
```

## Testing

Run the test suite:

```bash
ctest --test-dir build --output-on-failure
```

Tests are organized by component:
- `tests/parser/` - Lexer, parser, and type checker unit tests
- `tests/compiler/` - LLVM IR generation tests
- `tests/repl/` - REPL execution and input handling tests

## Documentation

- [Language Syntax](doc/syntax.md) - Complete syntax reference
