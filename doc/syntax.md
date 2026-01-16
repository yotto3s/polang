# Polang Language Syntax

Polang is a simple programming language with ML-inspired syntax and LLVM backend.

## Table of Contents

- [Types](#types)
- [Literals](#literals)
- [Variables](#variables)
- [Functions](#functions)
- [Expressions](#expressions)
- [Operators](#operators)
- [Comments](#comments)
- [Grammar Summary](#grammar-summary)

## Types

Polang supports two primitive types:

| Type     | Description                | Size    |
|----------|----------------------------|---------|
| `int`    | Signed integer             | 64-bit  |
| `double` | Floating-point number      | 64-bit  |

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

## Variables

### Variable Declaration

Variables are declared using the `let` keyword:

```
let x = 5
let y : int = 10
let pi : double = 3.14159
```

**Syntax:**
```
let <identifier> = <expression>
let <identifier> : <type> = <expression>
```

- When type is omitted, it defaults to `int`
- Variables must be initialized at declaration

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
let add (x : int) (y : int) : int = x + y
let square (n : int) = n * n
```

**Syntax:**
```
let <name> (<param> : <type>) ... : <return_type> = <expression>
let <name> (<param> : <type>) ... = <expression>
```

- Each parameter is enclosed in parentheses with its type
- Return type can be omitted (defaults to `int`)
- Function body is a single expression

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

## Expressions

Expressions can be:

- **Literals**: `42`, `3.14`
- **Identifiers**: `x`, `myVar`
- **Binary operations**: `a + b`, `x * y`
- **Comparisons**: `a == b`, `x < y`
- **Function calls**: `add(1, 2)`
- **Assignments**: `x = 5`
- **Parenthesized**: `(a + b) * c`

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

Currently, Polang does not support comments. All non-whitespace characters must be valid tokens.

## Grammar Summary

```ebnf
program     ::= statement*

statement   ::= var_decl
              | func_decl
              | expression

var_decl    ::= "let" identifier "=" expression
              | "let" identifier ":" type "=" expression

func_decl   ::= "let" identifier param+ ":" type "=" expression
              | "let" identifier param+ "=" expression

param       ::= "(" identifier ":" type ")"

expression  ::= identifier "=" expression
              | identifier "(" call_args ")"
              | identifier
              | numeric
              | expression binop expression
              | "(" expression ")"

call_args   ::= ε
              | expression ("," expression)*

binop       ::= "+" | "-" | "*" | "/"
              | "==" | "!=" | "<" | "<=" | ">" | ">="

identifier  ::= [a-zA-Z_][a-zA-Z0-9_]*

numeric     ::= integer | double

integer     ::= [0-9]+

double      ::= [0-9]+ "." [0-9]*

type        ::= "int" | "double"
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
let multiply (x : int) (y : int) : int = x * y
let result = multiply(6, 7)
```

### Comparison

```
let a = 5
let b = 10
let is_less = a < b
```

### Complex Expression

```
let compute (a : int) (b : int) (c : int) : int = (a + b) * c
let answer = compute(1, 2, 3)
```
