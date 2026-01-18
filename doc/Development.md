# Development Guide

This document describes the development workflow, code style guidelines, and tooling for the Polang project.

## Development Environment

### Docker (Recommended)

The project uses Docker for a consistent development environment:

```bash
# Start the development container
docker/docker_run.sh

# Run commands inside the container
docker/run_docker_command.sh <command> [options]

# Build the Docker image locally
docker/docker_build.sh
```

The container includes all required tools:
- GCC and Clang 20 compilers
- CMake, Bison, Flex
- LLVM 20 with MLIR
- clang-format, clang-tidy, clangd
- lcov, Python 3

### IDE Integration

#### clangd (Language Server)

Generate `compile_commands.json` for IDE support:

```bash
cmake -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_PREFIX_PATH="/usr/lib/llvm-20"
```

Run clangd with path mapping (inside container):

```bash
clangd --path-mappings=$(pwd)=/workspace/polang --enable-config
```

## Code Style

### General Guidelines

- Make variables and functions `const` whenever possible
- Mark functions `noexcept` when they don't throw exceptions
- Prefer `const` references for function parameters that aren't modified
- Add `[[nodiscard]]` to functions whose return value should not be ignored
- Use braces around all control flow statement bodies

### Smart Pointer Guidelines

**Avoid raw pointers for ownership.** Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) to manage dynamically allocated memory. This prevents memory leaks and makes ownership semantics explicit.

The AST uses `std::unique_ptr` for automatic memory management:

- **Use `std::unique_ptr`** for exclusive ownership of AST nodes
- **Use `std::make_unique`** for creating new nodes (exception-safe)
- **Use `.get()`** to obtain raw pointers for non-owning access (e.g., in visitors)
- **Use `std::move`** when transferring ownership between containers
- **Iterate with `const auto&`** over vectors of unique_ptr:
  ```cpp
  for (const auto& stmt : block->statements) {
      stmt->accept(visitor);
  }
  ```
- **Access members with `->`** through unique_ptr:
  ```cpp
  auto* varDecl = dynamic_cast<NVariableDeclaration*>(stmt.get());
  std::string name = varDecl->id->name;  // id is unique_ptr
  ```

**When raw pointers are acceptable:**
- Interfacing with C APIs or libraries that require raw pointers

**For non-owning access**, prefer references (`const T&` or `T&`) over raw pointers. References cannot be null and make non-owning semantics explicit.

**Avoid:**
- `new`/`delete` for memory management (use `std::make_unique` instead)
- Returning raw pointers that transfer ownership (return `std::unique_ptr`)
- Storing raw pointers in containers (use `std::vector<std::unique_ptr<T>>`)

### Naming Conventions (LLVM Style)

| Element | Case | Example |
|---------|------|---------|
| Classes/Structs | `CamelCase` | `TypeChecker`, `NBlock` |
| Enums | `CamelCase` | `ErrorSeverity` |
| Enum constants | `CamelCase` | `ImportKind::Module` |
| Functions/Methods | `lowerCamelCase` | `checkTypes()`, `getError()` |
| Variables | `lowerCamelCase` | `errorList`, `funcName` |
| Parameters | `lowerCamelCase` | `emitTypeVars`, `hasMore` |
| Members | `lowerCamelCase` | `inferredType`, `context` |
| Global/Static constants | `UPPER_CASE` | `TypeNames::INT` |
| Namespaces | `lowercase` | `polang` |

Configuration is defined in `.clang-tidy`.

## Code Formatting

### clang-format

Always format C/C++ files before committing (not `.l` or `.y` files):

```bash
# Format all source files
./scripts/run-clang-format.sh

# Check formatting without modifying (CI mode)
./scripts/run-clang-format.sh --check

# Format a specific file
clang-format -i <path/to/file.cpp>
```

Configuration is defined in `.clang-format`.

### clang-tidy

Run static analysis to catch issues:

```bash
# Run on all source files
./scripts/run-clang-tidy.sh

# Run with auto-fix (use with caution)
./scripts/run-clang-tidy.sh --fix

# Run on specific files
./scripts/run-clang-tidy.sh parser/src/ast_printer.cpp
```

Configuration is defined in `.clang-tidy`.

## Testing

### Running Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run specific test category
ctest --test-dir build -R "TypeCheck"

# Run lit tests only
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v build/tests/lit

# Run specific lit test category
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v build/tests/lit/MLIR
```

### Verifying Examples

After modifying the compiler, verify all examples still work:

```bash
for f in example/*.po; do echo "=== $(basename $f) ==="; ./build/bin/PolangRepl "$f"; done
```

Expected outputs:

| Example | Output |
|---------|--------|
| `closures.po` | `21 : i64` |
| `conditionals.po` | `10 : i64` |
| `factorial.po` | `120 : i64` |
| `fibonacci.po` | `5 : i64` |
| `functions.po` | `25 : i64` |
| `hello.po` | `7 : i64` |
| `let_expressions.po` | `16 : i64` |
| `mutability.po` | `23 : i64` |
| `types.po` | `84 : i64` |
| `variables.po` | `30 : i64` |

### Writing Lit Tests

Lit tests use FileCheck to verify compiler output:

```
; RUN: %polang_compiler --dump-ast %s | %FileCheck %s

; Test integer literal AST
; CHECK:      NBlock
; CHECK-NEXT: `-NExpressionStatement
; CHECK-NEXT:   `-NInteger 42
42
```

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

See `doc/Tests.md` for detailed test documentation.

## Documentation

When modifying code, update the relevant documentation:

| Change Type | Documentation |
|-------------|---------------|
| Language syntax | `doc/Syntax.md` |
| MLIR pipeline | `doc/Lowering.md` |
| Type system | `doc/TypeSystem.md` |
| Tests | `doc/Tests.md` |
| Build system | `doc/Building.md` |
| Architecture | `doc/Architecture.md` |

### Adding Lit Tests

When adding features or fixing bugs, add corresponding lit tests:

| Test Type | Directory | Flag |
|-----------|-----------|------|
| AST dump | `tests/lit/AST/` | `--dump-ast` |
| MLIR output | `tests/lit/MLIR/` | `--emit-mlir` |
| LLVM IR | `tests/lit/LLVMIR/` | (default) |
| Execution | `tests/lit/Execution/` | REPL |
| Errors | `tests/lit/Errors/` | `%not` |

## Continuous Integration

GitHub Actions runs on every push and pull request:

- **format-check**: Verifies clang-format compliance
- **lint**: Runs clang-tidy static analysis
- **build-and-test**: GCC/Clang × Debug/Release matrix
- **sanitizers**: AddressSanitizer and UndefinedBehaviorSanitizer
- **coverage**: Code coverage uploaded to Codecov

See `doc/CI_CD.md` for detailed CI documentation.

## Workflow Summary

1. **Make changes** to source code
2. **Format** with `./scripts/run-clang-format.sh`
3. **Lint** with `./scripts/run-clang-tidy.sh`
4. **Build** with `cmake --build build`
5. **Test** with `ctest --test-dir build --output-on-failure`
6. **Verify examples** work correctly
7. **Update documentation** as needed
8. **Commit** and push
