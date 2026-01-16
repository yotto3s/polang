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
| `hello.po` | Basic arithmetic | `7 : int` |
| `variables.po` | Variable declarations | `30 : int` |
| `functions.po` | Function declarations and calls | `25 : int` |
| `conditionals.po` | If-then-else expressions | `10 : int` |
| `let_expressions.po` | Let expressions with local bindings | `16 : int` |
| `types.po` | Different types (int, double, bool) | `84 : int` |
| `fibonacci.po` | Fibonacci sequence (5th number) | `5 : int` |
| `factorial.po` | Factorial of 5 | `120 : int` |
| `mutability.po` | Mutable variables and reassignment | `23 : int` |
| `closures.po` | Functions capturing variables from outer scope | `21 : int` |
