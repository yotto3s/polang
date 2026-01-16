# Polang Language Syntax

Polang is a simple programming language with ML-inspired syntax and LLVM backend.

## Table of Contents

- [Types](#types)
- [Literals](#literals)
- [Variables](#variables)
- [Functions](#functions)
- [Control Flow](#control-flow)
- [Expressions](#expressions)
- [Operators](#operators)
- [Comments](#comments)
- [Grammar Summary](#grammar-summary)

## Types

Polang supports three primitive types:

| Type     | Description                | Size    |
|----------|----------------------------|---------|
| `int`    | Signed integer             | 64-bit  |
| `double` | Floating-point number      | 64-bit  |
| `bool`   | Boolean value              | 1-bit   |

## Literals

### Integer Literals

Integer literals are sequences of digits:

```
0
42
12345
```

### Double Literals

Double literals are digits with a literal decimal point:

```
3.14
0.5
123.456
3.
```

Note: The decimal point is required. `3.` is valid (trailing digits are optional), but `3` without a decimal point is an integer.

### Boolean Literals

Boolean literals are `true` and `false`:

```
true
false
```

## Variables

### Variable Declaration

Variables are declared using the `let` keyword:

```
let x = 5           ; type inferred as int from literal
let y = 3.14        ; type inferred as double from literal
let z = true        ; type inferred as bool from literal
let w : int = 10    ; explicit type annotation
```

**Syntax:**
```
let <identifier> = <expression>
let <identifier> : <type> = <expression>
```

- When type is omitted, it is **inferred from the initializer expression**
- Variables must be initialized at declaration
- **No implicit type conversion**: `let x: double = 42` is an error (must write `42.0`)

### Variable Assignment

Existing variables can be reassigned:

```
let x = 5
x = 10
```

## Functions

### Function Declaration

Functions are declared using `let` with parameter lists:

```
let add(x: int, y: int): int = x + y   ; explicit return type
let square(n: int) = n * n              ; return type inferred as int
let half(x: double) = x / 2.0           ; return type inferred as double
```

**Syntax:**
```
let <name>(<param>: <type>, ...): <return_type> = <expression>
let <name>(<param>: <type>, ...) = <expression>
```

- Parameters are comma-separated within parentheses
- Each parameter requires a type annotation
- Return type can be omitted and will be **inferred from the body expression**
- Function body is a single expression
- **No implicit type conversion**: return type annotation must match body type exactly

### Function Calls

Functions are called with arguments in parentheses:

```
add(1, 2)
square(5)
print()
```

**Syntax:**
```
<function_name>(<arg1>, <arg2>, ...)
<function_name>()
```

## Control Flow

### If Expression

Polang supports conditional expressions using `if`/`then`/`else`:

```
if x > 0 then 1 else 0
if a == b then a + 1 else b + 1
```

**Syntax:**
```
if <condition> then <then_expr> else <else_expr>
```

- The condition must be a boolean expression (comparison or boolean literal)
- Both `then` and `else` branches are required
- If-expressions return a value and can be used anywhere an expression is expected
- If-expressions can be nested

**Examples:**

```
let max(a: int, b: int): int = if a > b then a else b
let abs(x: int): int = if x < 0 then 0 - x else x
let sign(n: int): int = if n > 0 then 1 else if n < 0 then 0 - 1 else 0
```

### Let Expression

Let-expressions introduce local variable bindings that are only visible within the body expression:

```
let x = 1 in x + 1
let x = 1 and y = 2 in x + y
let x : int = 1 and y : double = 2.0 in x
```

**Syntax:**
```
let <binding> (and <binding>)* in <expression>
```

Where `<binding>` is:
```
<identifier> = <expression>
<identifier> : <type> = <expression>
```

- Bindings are only visible within the body expression
- Multiple bindings are separated by `and`
- Each binding can optionally have a type annotation (defaults to `int`)
- The entire let-expression evaluates to the value of the body expression

**Examples:**

```
let a = 10 and b = 20 in a + b
let x = 5 in let y = x + 1 in y * 2
let sum(a: int, b: int): int = let result = a + b in result
```

## Expressions

Expressions can be:

- **Literals**: `42`, `3.14`, `true`, `false`
- **Identifiers**: `x`, `myVar`
- **Binary operations**: `a + b`, `x * y`
- **Comparisons**: `a == b`, `x < y` (return bool)
- **Function calls**: `add(1, 2)`
- **Assignments**: `x = 5`
- **Parenthesized**: `(a + b) * c`
- **If-expressions**: `if x > 0 then x else 0`
- **Let-expressions**: `let x = 1 in x + 1`

## Operators

### Arithmetic Operators

| Operator | Description    | Example   |
|----------|----------------|-----------|
| `+`      | Addition       | `a + b`   |
| `-`      | Subtraction    | `a - b`   |
| `*`      | Multiplication | `a * b`   |
| `/`      | Division       | `a / b`   |

### Comparison Operators

| Operator | Description              | Example   |
|----------|--------------------------|-----------|
| `==`     | Equal                    | `a == b`  |
| `!=`     | Not equal                | `a != b`  |
| `<`      | Less than                | `a < b`   |
| `<=`     | Less than or equal       | `a <= b`  |
| `>`      | Greater than             | `a > b`   |
| `>=`     | Greater than or equal    | `a >= b`  |

### Assignment Operator

| Operator | Description | Example  |
|----------|-------------|----------|
| `=`      | Assignment  | `x = 5`  |

### Operator Precedence

From highest to lowest:

1. `*`, `/` (multiplication, division)
2. `+`, `-` (addition, subtraction)
3. `==`, `!=`, `<`, `<=`, `>`, `>=` (comparisons)
4. `=` (assignment, right-associative)

Parentheses can be used to override precedence:

```
(1 + 2) * 3    // 9, not 7
```

## Comments

Polang supports single-line comments using the semicolon (`;`), following Lisp-style syntax:

```
; This is a comment
let x = 5  ; inline comment after code
; Comments extend to the end of the line
```

Comments are ignored by the parser and do not affect program execution. A file containing only comments is valid (produces an empty program).

## Grammar Summary

```ebnf
program     ::= statement*

statement   ::= var_decl
              | func_decl
              | expression

var_decl    ::= "let" identifier "=" expression
              | "let" identifier ":" type "=" expression

func_decl   ::= "let" identifier "(" param_list ")" ":" type "=" expression
              | "let" identifier "(" param_list ")" "=" expression
              | "let" identifier "()" ":" type "=" expression
              | "let" identifier "()" "=" expression

param_list  ::= param ("," param)*

param       ::= identifier ":" type

expression  ::= identifier "=" expression
              | identifier "(" call_args ")"
              | identifier
              | numeric
              | boolean
              | expression binop expression
              | "(" expression ")"
              | "if" expression "then" expression "else" expression
              | "let" let_bindings "in" expression

call_args   ::= ε
              | expression ("," expression)*

let_bindings ::= binding ("and" binding)*

binding     ::= identifier "=" expression
              | identifier ":" type "=" expression

binop       ::= "+" | "-" | "*" | "/"
              | "==" | "!=" | "<" | "<=" | ">" | ">="

identifier  ::= [a-zA-Z_][a-zA-Z0-9_]*

numeric     ::= integer | double

integer     ::= [0-9]+

double      ::= [0-9]+ "." [0-9]*

boolean     ::= "true" | "false"

type        ::= "int" | "double" | "bool"

comment     ::= ";" [^\n]*
```

## Examples

### Simple Variable

```
let x = 42
```

### Arithmetic Expression

```
let a = 10
let b = 20
let sum = a + b
```

### Function Definition and Call

```
let multiply(x: int, y: int): int = x * y
let result = multiply(6, 7)
```

### Comparison

```
let a = 5
let b = 10
let is_less : bool = a < b
```

### Complex Expression

```
let compute(a: int, b: int, c: int): int = (a + b) * c
let answer = compute(1, 2, 3)
```

### If Expression

```
let max(a: int, b: int): int = if a > b then a else b
let larger = max(10, 20)
```
