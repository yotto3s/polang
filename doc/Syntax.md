# Polang Language Syntax

Polang is a simple functional programming language with ML-inspired syntax and LLVM backend.

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

### Index Types

| Type    | Description                    | Size               |
|---------|--------------------------------|---------------------|
| `isize` | Signed index (pointer width)   | Platform-dependent  |
| `usize` | Unsigned index (pointer width) | Platform-dependent  |

Index types map to the platform-native pointer width. `usize` is intended for array indexing (Phase 2). Both types support explicit casting with `as`.

### Boolean Type

| Type   | Description   | Size   |
|--------|---------------|--------|
| `bool` | Boolean value | 1-bit  |

### Unit Type

| Type   | Description                              | Size   |
|--------|------------------------------------------|--------|
| `()`   | Unit type with a single value `()`       | 0-bit  |

The unit type `()` represents the absence of a meaningful value. It is used in function type signatures to indicate zero-parameter functions (e.g., `() -> i64`).

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

Variables are declared by binding names to expressions. All variables in Polang are **immutable**:

```
x = 5           (* type inferred as i64 *)
y = 3.14        (* type inferred as f64 *)
z = true        (* type inferred as bool *)
```

An optional **type signature** can be placed on the line before the definition:

```
w : i64
w = 10          (* explicit type via separate type signature *)
```

**Syntax:**
```
<identifier> = <expression>
```

With optional type signature on a preceding line:
```
<identifier> : <type>
<identifier> = <expression>
```

- When no type signature is provided, the type is **inferred from the initializer expression**
- Variables must be initialized at declaration
- **No implicit type conversion**: a variable with type signature `i64` cannot be initialized with `42.0`
- Variables cannot be reassigned after declaration
- Type signatures for top-level definitions are recommended; the compiler warns if they are missing

## Functions

### Function Declaration

Functions are declared by binding a name with a parameter list to an expression. An optional **type signature** on a preceding line specifies parameter and return types using arrow notation:

```
add : i64 * i64 -> i64                  (* type signature: two i64 params, returns i64 *)
add(x, y) = x + y                      (* definition *)

square(n) = n * n                       (* no type signature; types inferred *)
double(x) = x * 2                      (* parameter type inferred from body (i64) *)
half(x) = x / 2.0                      (* parameter type inferred as f64 *)
```

**Syntax:**

With type signature (recommended):
```
<name> : <type_expr>
<name>(<param>, ...) = <expression>
```

Without type signature (types inferred):
```
<name>(<param>, ...) = <expression>
```

**Type signature syntax:**
- Single parameter: `name : param_type -> return_type`
- Multiple parameters: `name : type1 * type2 -> return_type`
- No parameters: `name : () -> return_type`
- Arrow `->` is right-associative
- `*` (product) binds tighter than `->`

- Parameters are comma-separated within parentheses
- Parameter types come from the type signature; when no signature is provided, types are **inferred from usage**
- Return type comes from the type signature; when omitted, it is **inferred from the body expression**
- Function body is a single expression
- **No implicit type conversion**: type signatures must match inferred types exactly
- The compiler warns if a top-level function definition is missing a type signature

### Parameter Type Inference

Polang uses Hindley-Milner style type inference to determine parameter types. When a parameter type is omitted, Polang infers it from:

1. **Local usage** - How the parameter is used in the function body
2. **Call-site inference** - The types of arguments passed at call sites (polymorphic inference)

**Local inference examples:**

```
double(x) = x * 2       (* x inferred as i64 (from * 2) *)
half(x) = x / 2.0       (* x inferred as f64 (from / 2.0) *)
is_zero(x) = x == 0     (* x inferred as i64 (from == 0) *)
add(x, y) = x + y       (* both inferred from usage *)
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
identity(x) = x         (* x is polymorphic (type variable) *)
identity(42)             (* x inferred as i64 from call site *)
```

```
unused(x) = 42           (* x is polymorphic (type variable) *)
unused(1)                (* x inferred as i64 from call site *)
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
max : i64 * i64 -> i64
max(a, b) = if a > b then a else b

abs : i64 -> i64
abs(x) = if x < 0 then 0 - x else x

sign : i64 -> i64
sign(n) = if n > 0 then 1 else if n < 0 then 0 - 1 else 0
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

Where `<binding>` can be a variable binding:
```
<identifier> = <expression>
<identifier> : <type> = <expression>
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
(* Simple variable bindings *)
let a = 10 and b = 20 in a + b
let x = 5 in let y = x + 1 in y * 2

(* Function binding in let expression *)
let f(x: i64): i64 = x + 1 in f(5)

(* Multiple function bindings *)
let square(n: i64): i64 = n * n and cube(n: i64): i64 = n * n * n in square(3) + cube(2)

(* Mixed variable and function bindings *)
let x = 10 and double(y: i64): i64 = y * 2 in double(x)

(* Function with inferred return type *)
let inc(n: i64) = n + 1 in inc(41)
```

### Variable Capture (Closures)

Functions can capture variables from their enclosing scope:

```
x = 10
f() = x + 1   (* f captures x *)
f()            (* returns 11 *)
```

**Capture Semantics:**
- Variables are captured **by value** at call time
- Captured variables are passed as implicit extra parameters

**Examples:**

```
(* Simple capture *)
multiplier = 3
scale(n) = n * multiplier
scale(10)  (* returns 30 *)

(* Capture in let expression *)
result =
  let base = 100 and
      add(x: i64) = base + x
  in add(5)  (* returns 105 *)

(* Multiple captures *)
a = 1
b = 2
sum() = a + b
sum()  (* returns 3 *)
```

## Expressions

Expressions can be:

- **Literals**: `42`, `3.14`, `true`, `false`
- **Identifiers**: `x`, `myVar`
- **Binary operations**: `a + b`, `x * y`
- **Comparisons**: `a == b`, `x < y` (return bool)
- **Type conversions**: `x as i32`, `3.14 as i64`
- **Function calls**: `add(1, 2)`
- **Parenthesized**: `(a + b) * c`
- **If-expressions**: `if x > 0 then x else 0`
- **Let-expressions**: `let x = 1 in x + 1`

## Operators

### Unary Operators

Polang supports the following unary operators:

| Operator | Description   | Example |
|----------|---------------|---------|
| `-`      | Negation      | `-x`    |
| `!`      | Logical not   | `!flag` |

**Unary negation (`-`)** computes the arithmetic negation of its operand. The operand must be a numeric type (integer or float). The result has the same type as the operand.

**Logical not (`!`)** performs logical negation. The operand must be of type `bool`. The result is `false` if the operand is `true`, and `true` if the operand is `false`.

### Arithmetic Operators

| Operator | Description    | Example   |
|----------|----------------|-----------|
| `+`      | Addition       | `a + b`   |
| `-`      | Subtraction    | `a - b`   |
| `*`      | Multiplication | `a * b`   |
| `/`      | Division       | `a / b`   |
| `%`      | Modulo (remainder) | `a % b`   |

The modulo operator `%` computes the remainder of integer division (truncated division). The result has the same sign as the dividend. Float operands are not permitted with `%`; the program is ill-formed. Integer division and remainder by zero is undefined behavior.

Examples:
```
17 % 5     (* 2 *)
20 % 10    (* 0 *)
10 % 3     (* 1 *)
```

### Comparison Operators

| Operator | Description              | Example   |
|----------|--------------------------|-----------|
| `==`     | Equal                    | `a == b`  |
| `!=`     | Not equal                | `a != b`  |
| `<`      | Less than                | `a < b`   |
| `<=`     | Less than or equal       | `a <= b`  |
| `>`      | Greater than             | `a > b`   |
| `>=`     | Greater than or equal    | `a >= b`  |

### Type Conversion Operator

| Operator | Description                    | Example        | Returns          |
|----------|--------------------------------|----------------|------------------|
| `as`     | Explicit type conversion       | `x as i32`     | Converted value  |

The `as` operator converts a value from one numeric type to another. Only numeric-to-numeric conversions are allowed; boolean conversions are not permitted.

```
a : i64
a = 1000
b : i32
b = a as i32           (* narrow i64 to i32 *)
c : f64
c = a as f64           (* convert integer to float *)
d : i32
d = 3.7 as i32         (* convert float to integer (truncates to 3) *)
```

See [Type Conversions](TypeSystem.md#type-conversions) for detailed conversion semantics.

### Operator Precedence

Operators are listed from highest to lowest precedence:

| Precedence | Operators                           | Associativity   |
|------------|-------------------------------------|-----------------|
| 9          | `.` (member access)                 | Left            |
| 8          | Unary `-`, `!`                      | Right (prefix)  |
| 7          | `as` (type conversion)              | Left            |
| 6          | `*`, `/`, `%`                       | Left            |
| 5          | `+`, `-`                            | Left            |
| 4          | `==`, `!=`, `<`, `<=`, `>`, `>=`    | Non-associative |
| 3          | `&&`                                | Left            |
| 2          | `\|\|`                              | Left            |
| 1          | `if`/`then`/`else`, `let`/`in`      | Right           |

Examples:

```
-10 + 5             (* evaluated as: (-10) + 5 = -5 *)
!false && true      (* evaluated as: (!false) && true = true *)
a > 0 && a < 10 || b == 0    (* evaluated as: ((a > 0) && (a < 10)) || (b == 0) *)
10 % 3 + 1          (* evaluated as: (10 % 3) + 1 = 2 *)
```

Comparison operators are non-associative, meaning expressions like `a < b < c` are syntax errors and must use explicit parentheses:

```
a < b && b < c      (* correct *)
(a < b) < c         (* syntax error: can't compare bool and integer *)
```

## Comments

Polang uses OCaml-style block comments with `(* ... *)` delimiters:

```
(* This is a comment *)
x = 5  (* inline comment after code *)

(* Comments can span
   multiple lines *)

(* Comments can be nested: (* inner comment *) still in outer comment *)
```

Comments support arbitrary nesting, so `(* outer (* inner *) outer *)` is valid. An unterminated comment produces a syntax error at the position of the opening `(*`.

Comments are ignored by the parser and do not affect program execution. A file containing only comments is valid (produces an empty program).

## Modules

Polang supports a module system for organizing code into namespaces.

### Module Declaration

Modules are declared using the `module`/`endmodule` keywords with a Haskell-style export list:

```
module Math (add, PI)
  PI : f64
  PI = 3.14159

  add : i64 * i64 -> i64
  add(x, y) = x + y

  internal_helper(x) = x * 2  (* not exported *)
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
- Modules can contain type signatures, variables, functions, and nested modules

### Qualified Access

Module members are accessed using dot notation:

```
module Math (add, PI)
  PI = 3.14159
  add(x, y) = x + y
endmodule

Math.PI              (* access exported variable *)
Math.add(1, 2)       (* call exported function *)
```

### Import Statements

Import statements bring module symbols into the current scope:

**Import entire module:**
```
import Math                  (* use as Math.add, Math.PI *)
```

**Import with alias:**
```
import Math as M             (* use as M.add, M.PI *)
```

**Import specific items:**
```
from Math import add, PI     (* use directly as add, PI *)
from Math import add as plus (* use as plus instead of add *)
```

**Import all exports:**
```
from Math import *           (* import all exported symbols *)
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
  PI = 3.14159

  add : i64 * i64 -> i64
  add(x, y) = x + y

  mul : i64 * i64 -> i64
  mul(x, y) = x * y
endmodule

(* Using qualified access *)
Math.add(2, Math.mul(2, 3))  (* returns 8 *)

(* Using imports *)
from Math import add, mul
mul(2, add(1, 2))            (* returns 6 *)
```

**Private helpers:**
```
module Utils (process)
  (* Public function *)
  process : i64 -> i64
  process(x) = helper(x) + helper(x)

  (* Private helper (not exported) *)
  helper : i64 -> i64
  helper(x) = x * 2
endmodule

Utils.process(5)   (* returns 20 *)
Utils.helper(5)    (* ERROR: helper is not exported *)
```

**Nested modules:**
```
module Outer (Inner)
  module Inner (foo)
    foo : i64 -> i64
    foo(x) = x + 1
  endmodule
endmodule

Outer.Inner.foo(5)  (* returns 6 *)
```

## Grammar Summary

```ebnf
program     ::= statement*

statement   ::= type_signature
              | var_decl
              | func_decl
              | module_decl
              | import_stmt
              | expression

type_signature ::= identifier ":" type_expr

var_decl    ::= identifier "=" expression

func_decl   ::= identifier "(" param_list ")" "=" expression
              | identifier "()" "=" expression

module_decl ::= "module" identifier "(" ident_list ")" module_body "endmodule"
              | "module" identifier module_body "endmodule"

module_body ::= (type_signature | var_decl | func_decl | module_decl)*

import_stmt ::= "import" qualified_name
              | "import" qualified_name "as" identifier
              | "from" qualified_name "import" import_items
              | "from" qualified_name "import" "*"

import_items ::= identifier ("as" identifier)? ("," identifier ("as" identifier)?)*

qualified_name ::= identifier ("." identifier)*

ident_list  ::= identifier ("," identifier)*

param_list  ::= param ("," param)*

param       ::= identifier

expression  ::= qualified_name "(" call_args ")"
              | identifier "(" call_args ")"
              | qualified_name
              | identifier
              | numeric
              | boolean
              | expression binop expression
              | expression "as" type
              | "(" expression ")"
              | "if" expression "then" expression "else" expression
              | "let" let_bindings "in" expression

call_args   ::= ε
              | expression ("," expression)*

let_bindings ::= let_binding ("and" let_binding)*

let_binding ::= identifier "=" expression
              | identifier ":" type "=" expression
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

type        ::= base_type

type_expr    ::= "forall" type_var_list "." type_expr   (* quantified type *)
               | type_product "->" type_expr            (* right-associative *)
               | type_product
type_var_list ::= type_var_decl { "," type_var_decl }
type_var_decl ::= typevar                 (* unconstrained: 'a *)
                | typevar ":" identifier  (* constrained: 'a:Numeric *)
typevar      ::= "'" [a-z] [a-zA-Z0-9_]*
type_product ::= type_atom "*" type_product   (* `*` binds tighter than `->` *)
               | type_atom
type_atom    ::= type
               | typevar
               | "()"
               | "(" type_expr ")"

base_type   ::= "i8" | "i16" | "i32" | "i64"
              | "u8" | "u16" | "u32" | "u64"
              | "f32" | "f64"
              | "isize" | "usize"
              | "bool"
              | "()"

comment      ::= "(*" comment_body "*)"
comment_body ::= { any_char | comment }    (* nested comments allowed *)
any_char     ::= ? any character other than "(*", "*)", or EOF ?
```

## Examples

### Simple Variable

```
x = 42
```

### Arithmetic Expression

```
a = 10
b = 20
sum = a + b
```

### Function Definition and Call

```
multiply : i64 * i64 -> i64
multiply(x, y) = x * y

result = multiply(6, 7)
```

### Comparison

```
a = 5
b = 10

is_less : bool
is_less = a < b
```

### Complex Expression

```
compute : i64 * i64 * i64 -> i64
compute(a, b, c) = (a + b) * c

answer = compute(1, 2, 3)
```

### If Expression

```
max : i64 * i64 -> i64
max(a, b) = if a > b then a else b

larger = max(10, 20)
```

### Type Conversions

```
(* Integer narrowing (truncates) *)
big : i64
big = 1000
small : i8
small = big as i8        (* small = -24 (1000 mod 256, interpreted as signed) *)

(* Integer to float *)
n : i32
n = 42
f : f64
f = n as f64            (* f = 42.0 *)

(* Float to integer (truncates toward zero, saturates at bounds) *)
pi : f64
pi = 3.14159
rounded : i32
rounded = pi as i32     (* rounded = 3 *)

(* Mixed arithmetic with conversions *)
a : i32
a = 10
b : i64
b = 20
sum : i64
sum = a as i64 + b      (* convert a to i64 before adding *)

(* Index type conversions *)
idx : isize
idx = 42 as isize       (* convert integer to isize *)
n2 : i64
n2 = idx as i64         (* convert isize back to integer *)
uidx : usize
uidx = 10 as usize      (* convert integer to usize *)
```
