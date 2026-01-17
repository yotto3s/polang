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
- [Modules](#modules)
- [Grammar Summary](#grammar-summary)

## Types

Polang supports a variety of numeric types with explicit width and signedness:

### Integer Types

| Type  | Description                | Size    |
|-------|----------------------------|---------|
| `i8`  | Signed 8-bit integer       | 8-bit   |
| `i16` | Signed 16-bit integer      | 16-bit  |
| `i32` | Signed 32-bit integer      | 32-bit  |
| `i64` | Signed 64-bit integer      | 64-bit  |
| `u8`  | Unsigned 8-bit integer     | 8-bit   |
| `u16` | Unsigned 16-bit integer    | 16-bit  |
| `u32` | Unsigned 32-bit integer    | 32-bit  |
| `u64` | Unsigned 64-bit integer    | 64-bit  |

### Floating-Point Types

| Type  | Description                | Size    |
|-------|----------------------------|---------|
| `f32` | Single-precision float     | 32-bit  |
| `f64` | Double-precision float     | 64-bit  |

### Boolean Type

| Type   | Description   | Size   |
|--------|---------------|--------|
| `bool` | Boolean value | 1-bit  |

### Legacy Type Aliases

For compatibility, the following aliases are supported:

| Alias    | Maps To |
|----------|---------|
| `int`    | `i64`   |
| `double` | `f64`   |

### Default Literal Types

- Integer literals (e.g., `42`) default to `i64`
- Float literals (e.g., `3.14`) default to `f64`

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

Variables are declared using the `let` keyword. By default, variables are **immutable**:

```
let x = 5           ; immutable, type inferred as int
let y = 3.14        ; immutable, type inferred as double
let z = true        ; immutable, type inferred as bool
let w : int = 10    ; immutable, explicit type annotation
```

**Syntax:**
```
let <identifier> = <expression>
let <identifier> : <type> = <expression>
```

- When type is omitted, it is **inferred from the initializer expression**
- Variables must be initialized at declaration
- **No implicit type conversion**: `let x: double = 42` is an error (must write `42.0`)

### Mutable Variables

To declare a mutable variable that can be reassigned, use `let mut`:

```
let mut x = 5       ; mutable, type inferred as int
let mut y : int = 10  ; mutable, explicit type annotation
```

**Syntax:**
```
let mut <identifier> = <expression>
let mut <identifier> : <type> = <expression>
```

### Variable Reassignment

Only mutable variables can be reassigned using the `<-` operator:

```
let mut x = 5
x <- 10             ; OK: x is mutable

let y = 5
y <- 10             ; ERROR: cannot reassign immutable variable
```

**Assignment as Expression:**

The assignment operator `<-` returns the assigned value, making it an expression:

```
let mut x = 0
x <- 10             ; evaluates to 10

; Chained assignment (right-associative)
let mut a = 0
let mut b = 0
a <- b <- 5         ; assigns 5 to both a and b, evaluates to 5
```

**Note:** The `=` operator is used only for initial binding (declaration). The `<-` operator is used for reassignment (mutation).

## Functions

### Function Declaration

Functions are declared using `let` with parameter lists:

```
let add(x: i64, y: i64): i64 = x + y   ; explicit types
let square(n: i64) = n * n              ; return type inferred as i64
let double(x) = x * 2                   ; parameter type inferred from body (i64)
let half(x) = x / 2.0                   ; parameter type inferred as f64
```

**Syntax:**
```
let <name>(<param>: <type>, ...): <return_type> = <expression>
let <name>(<param>: <type>, ...) = <expression>
let <name>(<param>, ...) = <expression>    ; types inferred
```

- Parameters are comma-separated within parentheses
- Parameter type annotations are optional; when omitted, types are **inferred from usage**
- Return type can be omitted and will be **inferred from the body expression**
- Function body is a single expression
- **No implicit type conversion**: explicit type annotations must match inferred types exactly

### Parameter Type Inference

Polang uses Hindley-Milner style type inference to determine parameter types. When a parameter type is omitted, Polang infers it from:

1. **Local usage** - How the parameter is used in the function body
2. **Call-site inference** - The types of arguments passed at call sites (polymorphic inference)

**Local inference examples:**

```
let double(x) = x * 2       ; x inferred as i64 (from * 2)
let half(x) = x / 2.0       ; x inferred as f64 (from / 2.0)
let is_zero(x) = x == 0     ; x inferred as i64 (from == 0)
let add(x: i64, y) = x + y  ; y inferred as i64 (from + x)
```

**Local inference rules:**
- `x + 1` or `x * 2` (integer literal) → x is `i64`
- `x + 1.0` or `x / 2.0` (float literal) → x is `f64`
- `if x then ...` (used as condition) → x is `bool`
- `x + y` where y has known type → x has same type
- `f(x)` where f expects a type → x has that type

**Polymorphic call-site inference:**

When a parameter's type cannot be determined from local usage, Polang infers it from the call site:

```
let identity(x) = x         ; x is polymorphic (type variable)
identity(42)                ; x inferred as i64 from call site
```

```
let unused(x) = 42          ; x is polymorphic (type variable)
unused(1)                   ; x inferred as i64 from call site
```

This enables polymorphic functions where the same function definition can work with different types based on how it's called. The type inference happens at the MLIR level using a unification-based algorithm.

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
let max(a: i64, b: i64): i64 = if a > b then a else b
let abs(x: i64): i64 = if x < 0 then 0 - x else x
let sign(n: i64): i64 = if n > 0 then 1 else if n < 0 then 0 - 1 else 0
```

### Let Expression

Let-expressions introduce local bindings (variables or functions) that are only visible within the body expression:

```
let x = 1 in x + 1
let x = 1 and y = 2 in x + y
let x : int = 1 and y : double = 2.0 in x
```

**Syntax:**
```
let <binding> (and <binding>)* in <expression>
```

Where `<binding>` can be a variable binding (immutable by default):
```
<identifier> = <expression>
<identifier> : <type> = <expression>
mut <identifier> = <expression>
mut <identifier> : <type> = <expression>
```

Or a function binding:
```
<identifier>(<param>: <type>, ...): <return_type> = <expression>
<identifier>(<param>: <type>, ...) = <expression>
```

- Bindings are only visible within the body expression
- Multiple bindings are separated by `and`
- Bindings can be variables or functions mixed together
- Each variable binding can optionally have a type annotation
- Each function binding can optionally have a return type annotation (inferred if omitted)
- The entire let-expression evaluates to the value of the body expression

**Examples:**

```
; Simple variable bindings
let a = 10 and b = 20 in a + b
let x = 5 in let y = x + 1 in y * 2

; Function binding in let expression
let f(x: i64): i64 = x + 1 in f(5)

; Multiple function bindings
let square(n: i64): i64 = n * n and cube(n: i64): i64 = n * n * n in square(3) + cube(2)

; Mixed variable and function bindings
let x = 10 and double(y: i64): i64 = y * 2 in double(x)

; Function with inferred return type
let inc(n: i64) = n + 1 in inc(41)
```

### Variable Capture (Closures)

Functions can capture variables from their enclosing scope:

```
let x = 10
let f() = x + 1   ; f captures x
f()               ; returns 11
```

**Capture Semantics:**
- Variables are captured **by value** at call time
- Captured variables are passed as implicit extra parameters
- Both mutable and immutable variables can be captured
- Mutations to captured variables inside the function do not affect the outer variable

**Examples:**

```
; Simple capture
let multiplier = 3
let scale(n: i64) = n * multiplier
scale(10)  ; returns 30

; Capture in let expression
let result =
  let base = 100 and
      add(x: i64) = base + x
  in add(5)  ; returns 105

; Multiple captures
let a = 1
let b = 2
let sum() = a + b
sum()  ; returns 3
```

## Expressions

Expressions can be:

- **Literals**: `42`, `3.14`, `true`, `false`
- **Identifiers**: `x`, `myVar`
- **Binary operations**: `a + b`, `x * y`
- **Comparisons**: `a == b`, `x < y` (return bool)
- **Function calls**: `add(1, 2)`
- **Reassignments**: `x <- 5` (for mutable variables only)
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

### Reassignment Operator

| Operator | Description                           | Example   | Returns        |
|----------|---------------------------------------|-----------|----------------|
| `<-`     | Reassignment (mutable variables only) | `x <- 5`  | Assigned value |

The reassignment operator returns the assigned value, allowing chained assignments like `a <- b <- 5`.

**Note:** The `=` operator is used only for initial binding in declarations (`let x = 5`).

### Operator Precedence

From highest to lowest:

1. `*`, `/` (multiplication, division)
2. `+`, `-` (addition, subtraction)
3. `==`, `!=`, `<`, `<=`, `>`, `>=` (comparisons)
4. `<-` (reassignment, right-associative)

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

## Modules

Polang supports a module system for organizing code into namespaces.

### Module Declaration

Modules are declared using the `module`/`endmodule` keywords with a Haskell-style export list:

```
module Math (add, PI)
  let PI = 3.14159
  let add(x: i64, y: i64): i64 = x + y
  let internal_helper(x: i64): i64 = x * 2  ; not exported
endmodule
```

**Syntax:**
```
module <name> (<export1>, <export2>, ...)
  <declarations>
endmodule
```

- The export list in parentheses specifies which symbols are public
- Symbols not in the export list are private to the module
- A module without an export list has no public symbols
- Modules can contain variables, functions, and nested modules

### Qualified Access

Module members are accessed using dot notation:

```
module Math (add, PI)
  let PI = 3.14159
  let add(x: i64, y: i64): i64 = x + y
endmodule

Math.PI              ; access exported variable
Math.add(1, 2)       ; call exported function
```

### Import Statements

Import statements bring module symbols into the current scope:

**Import entire module:**
```
import Math                  ; use as Math.add, Math.PI
```

**Import with alias:**
```
import Math as M             ; use as M.add, M.PI
```

**Import specific items:**
```
from Math import add, PI     ; use directly as add, PI
from Math import add as plus ; use as plus instead of add
```

**Import all exports:**
```
from Math import *           ; import all exported symbols
```

**Syntax:**
```
import <module>
import <module> as <alias>
from <module> import <item1>, <item2>, ...
from <module> import <item> as <alias>, ...
from <module> import *
```

### Module Examples

**Basic module with function and variable:**
```
module Math (add, mul, PI)
  let PI = 3.14159
  let add(x: i64, y: i64): i64 = x + y
  let mul(x: i64, y: i64): i64 = x * y
endmodule

; Using qualified access
Math.add(2, Math.mul(2, 3))  ; returns 8

; Using imports
from Math import add, mul
mul(2, add(1, 2))            ; returns 6
```

**Private helpers:**
```
module Utils (process)
  ; Public function
  let process(x: i64): i64 = helper(x) + helper(x)

  ; Private helper (not exported)
  let helper(x: i64): i64 = x * 2
endmodule

Utils.process(5)   ; returns 20
Utils.helper(5)    ; ERROR: helper is not exported
```

**Nested modules:**
```
module Outer (Inner)
  module Inner (foo)
    let foo(x: i64): i64 = x + 1
  endmodule
endmodule

Outer.Inner.foo(5)  ; returns 6
```

## Grammar Summary

```ebnf
program     ::= statement*

statement   ::= var_decl
              | func_decl
              | module_decl
              | import_stmt
              | expression

var_decl    ::= "let" identifier "=" expression
              | "let" identifier ":" type "=" expression
              | "let" "mut" identifier "=" expression
              | "let" "mut" identifier ":" type "=" expression

func_decl   ::= "let" identifier "(" param_list ")" ":" type "=" expression
              | "let" identifier "(" param_list ")" "=" expression
              | "let" identifier "()" ":" type "=" expression
              | "let" identifier "()" "=" expression

module_decl ::= "module" identifier "(" ident_list ")" module_body "endmodule"
              | "module" identifier module_body "endmodule"

module_body ::= (var_decl | func_decl | module_decl)*

import_stmt ::= "import" qualified_name
              | "import" qualified_name "as" identifier
              | "from" qualified_name "import" import_items
              | "from" qualified_name "import" "*"

import_items ::= identifier ("as" identifier)? ("," identifier ("as" identifier)?)*

qualified_name ::= identifier ("." identifier)*

ident_list  ::= identifier ("," identifier)*

param_list  ::= param ("," param)*

param       ::= identifier ":" type
              | identifier

expression  ::= identifier "<-" expression
              | qualified_name "(" call_args ")"
              | identifier "(" call_args ")"
              | qualified_name
              | identifier
              | numeric
              | boolean
              | expression binop expression
              | "(" expression ")"
              | "if" expression "then" expression "else" expression
              | "let" let_bindings "in" expression

call_args   ::= ε
              | expression ("," expression)*

let_bindings ::= let_binding ("and" let_binding)*

let_binding ::= identifier "=" expression
              | identifier ":" type "=" expression
              | "mut" identifier "=" expression
              | "mut" identifier ":" type "=" expression
              | identifier "(" param_list ")" ":" type "=" expression
              | identifier "(" param_list ")" "=" expression
              | identifier "()" ":" type "=" expression
              | identifier "()" "=" expression

binop       ::= "+" | "-" | "*" | "/"
              | "==" | "!=" | "<" | "<=" | ">" | ">="

identifier  ::= [a-zA-Z_][a-zA-Z0-9_]*

numeric     ::= integer | double

integer     ::= [0-9]+

double      ::= [0-9]+ "." [0-9]*

boolean     ::= "true" | "false"

type        ::= "i8" | "i16" | "i32" | "i64"
              | "u8" | "u16" | "u32" | "u64"
              | "f32" | "f64"
              | "int" | "double"   ; legacy aliases
              | "bool"

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
let multiply(x: i64, y: i64): i64 = x * y
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
let compute(a: i64, b: i64, c: i64): i64 = (a + b) * c
let answer = compute(1, 2, 3)
```

### If Expression

```
let max(a: i64, b: i64): i64 = if a > b then a else b
let larger = max(10, 20)
```

### Mutable Variables

```
let mut counter = 0
counter <- counter + 1
counter <- counter + 1
counter <- counter + 1
; counter is now 3

; Chained assignment
let mut x = 0
let mut y = 0
x <- y <- 10
; both x and y are 10
```
