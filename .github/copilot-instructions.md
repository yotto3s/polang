# Polang Code Review Guidelines

This document provides comprehensive review guidelines for GitHub Copilot when reviewing pull requests for the Polang project.

## Documentation References

Before reviewing, familiarize yourself with:

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Quick reference for commands and code style |
| `README.md` | Project overview |
| `doc/Architecture.md` | Project structure and MLIR lowering pipeline |
| `doc/Development.md` | Full code style guide and workflow |
| `doc/Syntax.md` | Language syntax reference |
| `doc/TypeSystem.md` | Type system and inference |
| `doc/Testing.md` | Test infrastructure, coverage, and CI/CD |
| `doc/Building.md` | Build instructions and dependencies |

---

## Code Style

### Naming Conventions (LLVM Style)

- [ ] **Types** (classes, structs, enums): `CamelCase` (e.g., `TypeChecker`, `NBlock`)
- [ ] **Functions/methods**: `lowerCamelCase` (e.g., `checkTypes()`, `getError()`)
- [ ] **Variables/parameters/members**: `lowerCamelCase` (e.g., `errorList`, `funcName`)
- [ ] **Global/static constants**: `UPPER_CASE` (e.g., `TypeNames::INT`)
- [ ] **Namespaces**: `lowercase` (e.g., `polang`)

### Formatting

- [ ] Braces required on **all** control flow bodies (even single-line `if`/`for`/`while`)
- [ ] Pointer alignment: left (`Type* var`, not `Type *var`)
- [ ] Code must pass `./scripts/run-clang-format.sh --check`

---

## C++ Best Practices

This section is **critical** for maintaining code quality. Each item includes rationale.

### const Correctness

- [ ] **Use `const` wherever possible**
  - Variables that don't change after initialization should be `const`
  - Prevents accidental modification and enables compiler optimizations
- [ ] **Use `const` references for function parameters**
  - `void process(const std::string& input)` instead of `void process(std::string input)`
  - Avoids unnecessary copies while preventing modification
- [ ] **Mark methods `const` when they don't modify object state**
  - Enables calling the method on const objects

### noexcept Specification

- [ ] **Mark functions `noexcept` when they don't throw exceptions**
  - Enables compiler optimizations (especially for move operations)
  - Documents the function's exception guarantee
  - Required for move constructors/assignment to enable efficient container operations

### [[nodiscard]] Attribute

- [ ] **Add `[[nodiscard]]` to functions with important return values**
  - Prevents bugs from ignoring error codes or computed values
  - Essential for factory functions, getters, and functions returning status/error codes
  - Example: `[[nodiscard]] bool isValid() const;`

### Smart Pointers and Memory Management

- [ ] **Use `std::unique_ptr` for exclusive ownership**
  - The AST uses `std::unique_ptr` for automatic memory management
  - Makes ownership semantics explicit
- [ ] **Use `std::make_unique` for creating objects** (not raw `new`)
  - Exception-safe: if construction fails, no memory leaks
  - Example: `auto node = std::make_unique<NBlock>();`
- [ ] **No raw `new`/`delete` for memory management**
  - Use smart pointers instead
  - Prevents memory leaks and dangling pointers
- [ ] **Use `.get()` for non-owning access to unique_ptr contents**
- [ ] **Use `std::move` when transferring ownership**
- [ ] **Prefer references over raw pointers for non-owning access**
  - References cannot be null, making semantics clearer

### AST Ownership Model

```cpp
// Correct: iterate with const auto&
for (const auto& stmt : block->statements) {
    stmt->accept(visitor);
}

// Correct: access members through unique_ptr
auto* varDecl = dynamic_cast<NVariableDeclaration*>(stmt.get());
std::string name = varDecl->id->name;  // id is unique_ptr
```

---

## Static Analysis

- [ ] **Code must pass clang-tidy** (`./scripts/run-clang-tidy.sh`)
  - Checks enabled: `bugprone-*`, `performance-*`, `modernize-*`, `readability-*`
  - Configuration in `.clang-tidy`
- [ ] **No magic numbers** (except 0, 1, 2, -1, and powers of 2)
  - Use named constants or enums for clarity
  - Example: `constexpr int MAX_ITERATIONS = 100;`

---

## Testing Requirements

This section is **critical** for maintaining test coverage and CI health.

### Lit Test Categories

When reviewing changes, ensure appropriate tests are added:

| Directory | Description | Flag |
|-----------|-------------|------|
| `tests/lit/AST/` | AST dump tests | `--dump-ast` |
| `tests/lit/MLIR/` | MLIR output tests | `--emit-mlir` |
| `tests/lit/LLVMIR/` | LLVM IR generation | (default) |
| `tests/lit/Execution/` | REPL execution | Uses PolangRepl |
| `tests/lit/Errors/` | Error message verification | `%not` |

### When to Add Each Test Type

- **AST tests**: Changes to lexer (`lexer.l`), parser (`parser.y`), or AST nodes (`node.hpp`)
- **MLIR tests**: Changes to Polang dialect or AST-to-MLIR lowering
- **LLVMIR tests**: Changes to MLIR-to-LLVM lowering
- **Execution tests**: Changes affecting runtime behavior
- **Error tests**: Changes to error detection or message formatting

### Expected Example Outputs

After any compiler changes, all examples must produce correct output:

| Example | Expected Output |
|---------|-----------------|
| `closures.po` | `21 : i64` |
| `conditionals.po` | `10 : i64` |
| `factorial.po` | `120 : i64` |
| `fibonacci.po` | `5 : i64` |
| `functions.po` | `25 : i64` |
| `hello.po` | `7 : i64` |
| `let_expressions.po` | `16 : i64` |
| `types.po` | `84 : i64` |
| `variables.po` | `30 : i64` |

### Coverage Targets

- **Lines**: ~86% (2314 of 2680 lines)
- **Functions**: ~92% (280 of 304 functions)

New code should maintain or improve these coverage levels.

### CI Configurations

All PRs must pass these CI jobs:

| Job | Description |
|-----|-------------|
| `format-check` | clang-format compliance |
| `lint` | clang-tidy static analysis |
| `build-and-test` | GCC/Clang x Debug/Release (4 configurations) |
| `sanitizers` | AddressSanitizer + UndefinedBehaviorSanitizer |
| `coverage` | Code coverage report |

**Build Matrix:**
- GCC Debug
- GCC Release
- Clang 20 Debug
- Clang 20 Release

---

## Documentation Requirements

When code changes, corresponding documentation must be updated:

| Code Change | Documentation to Update |
|-------------|------------------------|
| Lexer (`lexer.l`), parser (`parser.y`), AST nodes (`node.hpp`) | `doc/Syntax.md` |
| MLIR pipeline (passes, lowering) | `doc/Architecture.md` |
| Type system (type checker, inference) | `doc/TypeSystem.md` |
| Tests or CI configuration | `doc/Testing.md` |
| Build system (CMake) | `doc/Building.md` |
| Project structure or components | `doc/Architecture.md` |

---

## Review Checklist

Use this checklist for every PR review:

### Code Quality
- [ ] Naming follows LLVM conventions (CamelCase types, lowerCamelCase functions/variables)
- [ ] Braces on all control flow bodies
- [ ] `const` used wherever possible
- [ ] `noexcept` on non-throwing functions
- [ ] `[[nodiscard]]` on functions with important return values
- [ ] Smart pointers for ownership (no raw `new`/`delete`)
- [ ] `std::make_unique` used for object creation
- [ ] No magic numbers (use named constants)

### Static Analysis
- [ ] Code passes `./scripts/run-clang-format.sh --check`
- [ ] Code passes `./scripts/run-clang-tidy.sh`

### Testing
- [ ] Appropriate lit tests added (AST/MLIR/LLVMIR/Execution/Errors)
- [ ] GTest added for edge cases and error handling (if applicable)
- [ ] All example programs produce expected output
- [ ] Coverage maintained or improved

### Documentation
- [ ] Relevant documentation updated (see table above)
- [ ] Code comments added for non-obvious logic

### CI
- [ ] All CI jobs pass (format, lint, build matrix, sanitizers, coverage)
