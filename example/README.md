# Polang Examples

Example programs demonstrating Polang language features.

## Running Examples

```bash
# Using the REPL
cat example/hello.po | ./build/bin/PolangRepl

# Using the compiler (outputs LLVM IR)
cat example/hello.po | ./build/bin/PolangCompiler
```

## Examples

| File | Description | Output |
|------|-------------|--------|
| `hello.po` | Basic arithmetic | `7 : i64` |
| `variables.po` | Variable declarations | `30 : i64` |
| `functions.po` | Function declarations and calls | `25 : i64` |
| `conditionals.po` | If-then-else expressions | `10 : i64` |
| `let_expressions.po` | Let expressions with local bindings | `16 : i64` |
| `types.po` | Different types (i64, f64, bool) | `84 : i64` |
| `fibonacci.po` | Fibonacci sequence (5th number) | `5 : i64` |
| `factorial.po` | Factorial of 5 | `120 : i64` |
| `mutability.po` | Mutable variables and reassignment | `23 : i64` |
| `closures.po` | Functions capturing variables from outer scope | `21 : i64` |
