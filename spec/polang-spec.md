# The Polang Language Specification

Version: 0.1.0 (Draft)

## 1. Introduction

This is a reference manual for the Polang programming language.

Polang is a statically typed, functional programming language with ML-inspired syntax and an LLVM backend. It features Hindley-Milner type inference, polymorphic functions via monomorphization, and immutable variables.

This document specifies the syntax and semantics of all currently implemented language features. It serves as the authoritative reference for the language; the implementation must conform to this specification.

## 2. Notation

The syntax is specified using a variant of Extended Backus-Naur Form (EBNF):

    Syntax      = { Production } .
    Production  = production_name "=" Expression "." .
    Expression  = Term { "|" Term } .
    Term        = Factor { Factor } .
    Factor      = production_name
                | token
                | Group
                | Option
                | Repetition .
    Group       = "(" Expression ")" .
    Option      = "[" Expression "]" .
    Repetition  = "{" Expression "}" .

Productions are expressions constructed from terms and the following operators, in increasing precedence:

    |   alternation
    ()  grouping
    []  option (0 or 1 times)
    {}  repetition (0 or more times)

Lowercase production names are used to identify lexical (terminal) tokens. Non-terminals are in CamelCase. Literal tokens are enclosed in double quotes `""`.

The form `a ... b` represents the set of characters from `a` through `b` as alternatives.

## 3. Source Code Representation

### 3.1 Characters

Source code is a sequence of UTF-8 encoded Unicode characters. The implementation currently processes only the ASCII subset of Unicode for lexical tokens.

### 3.2 Letters and Digits

    letter        = "a" ... "z" | "A" ... "Z" | "_" .
    decimal_digit = "0" ... "9" .

### 3.3 Comments

Polang uses OCaml-style block comments delimited by `(*` and `*)`:

    Comment = "(*" CommentBody "*)" .
    CommentBody = { CommentChar | Comment } .
    CommentChar = /* any character except "(*" and "*)" */ .

Comments can be nested. An unterminated comment is a compile-time error reported at the position of the opening `(*`.

Comments are treated as whitespace and do not affect program semantics.

    (* This is a comment *)
    (* Comments (* can be (* nested *) arbitrarily *) deep *)

## 4. Lexical Elements

### 4.1 Tokens

Tokens form the vocabulary of the language. There are four classes: identifiers, keywords, literals, and operators/punctuation. Whitespace (spaces, tabs, newlines) is ignored except as it separates tokens.

### 4.2 Identifiers

Identifiers name variables, functions, and modules.

    identifier = letter { letter | decimal_digit } .

Examples:

    x
    myVar
    add_two
    _internal
    Math

### 4.3 Keywords

The following words are reserved and must not be used as identifiers:

    let     fun     in      and     if      then    else
    true    false   module  endmodule       import  from
    as      forall

### 4.4 Operators and Punctuation

The following character sequences represent operators and punctuation:

    +    -    *    /    %
    &+   &-   &*
    ==   !=   <    <=   >    >=
    &&   ||   !
    =    :    ->   .    ,
    (    )    {    }

### 4.5 Integer Literals

An integer literal is a sequence of decimal digits representing an integer value.

    integer_lit = decimal_digit { decimal_digit } .

Examples:

    0
    42
    12345

Integer literals have no inherent type. Their type is determined by context (see [§12.4 Literal Type Inference](#124-literal-type-inference)). When no type context is available, an integer literal has type `i64`.

### 4.6 Floating-Point Literals

A floating-point literal is a sequence of decimal digits with a decimal point.

    float_lit = decimal_digit { decimal_digit } "." { decimal_digit } .

The decimal point is required. Trailing digits after the decimal point are optional.

Examples:

    3.14
    0.5
    123.456
    3.

The literal `3.` is a valid floating-point literal (equivalent to `3.0`). The literal `3` without a decimal point is an integer literal, not a floating-point literal.

When no type context is available, a floating-point literal has type `f64`.

### 4.7 Boolean Literals

    bool_lit = "true" | "false" .

### 4.8 Unit Literal

The unit literal represents the sole value of the unit type:

    unit_lit = "(" ")" .

### 4.9 Type Variable Literals

Type variable literals appear in type expressions and use ML-style naming:

    typevar = "'" lower_letter { letter | decimal_digit } .
    lower_letter = "a" ... "z" .

Examples:

    'a
    'b
    'myType

## 5. Constants and Literals

### 5.1 Numeric Constants

Integer and floating-point literals are compile-time constants. The compiler evaluates literal values at compile time and checks that they fit within the declared type's range.

If a literal value does not fit in its target type, the program is ill-formed:

    x : i8
    x = 1000             (* ill-formed: 1000 does not fit in i8 *)

    y : u8
    y = -1               (* ill-formed: -1 does not fit in u8 *)

### 5.2 Boolean Constants

The literals `true` and `false` are the two values of type `bool`.

### 5.3 Unit Constant

The literal `()` is the sole value of the unit type `()`.

## 6. Types

### 6.1 Integer Types

Polang provides signed and unsigned integer types with explicit widths:

| Type    | Description              | Size    | Range                                    |
|---------|--------------------------|---------|------------------------------------------|
| `i8`    | Signed 8-bit integer     | 8-bit   | -128 to 127                              |
| `i16`   | Signed 16-bit integer    | 16-bit  | -32768 to 32767                          |
| `i32`   | Signed 32-bit integer    | 32-bit  | -2147483648 to 2147483647                |
| `i64`   | Signed 64-bit integer    | 64-bit  | -2^63 to 2^63-1                          |
| `u8`    | Unsigned 8-bit integer   | 8-bit   | 0 to 255                                 |
| `u16`   | Unsigned 16-bit integer  | 16-bit  | 0 to 65535                               |
| `u32`   | Unsigned 32-bit integer  | 32-bit  | 0 to 4294967295                          |
| `u64`   | Unsigned 64-bit integer  | 64-bit  | 0 to 2^64-1                              |

Integer arithmetic uses two's complement representation for signed types and standard binary representation for unsigned types.

Integer overflow (both signed and unsigned) shall cause the program to print a runtime error message including the source location and exit with a non-zero status. This applies to `+`, `-`, `*`, and unary `-`. Division overflow (the sole case being `MIN_INT / -1` for signed types) is also a runtime error.

### 6.2 Floating-Point Types

| Type  | Description             | Size    | Standard       |
|-------|-------------------------|---------|----------------|
| `f32` | Single-precision float  | 32-bit  | IEEE 754       |
| `f64` | Double-precision float  | 64-bit  | IEEE 754       |

Floating-point types conform to the IEEE 754 standard. Special values include positive and negative infinity and NaN (Not a Number).

### 6.3 Index Types

| Type    | Description                    | Size               |
|---------|--------------------------------|--------------------|
| `isize` | Signed index (pointer width)   | Platform-dependent |
| `usize` | Unsigned index (pointer width) | Platform-dependent |

Index types map to the platform-native pointer width. On 64-bit platforms, both `isize` and `usize` are 64 bits wide.

### 6.4 Boolean Type

| Type   | Description   | Size   |
|--------|---------------|--------|
| `bool` | Boolean value | 1-bit  |

The boolean type has exactly two values: `true` and `false`.

### 6.5 Unit Type

| Type   | Description                        | Size   |
|--------|------------------------------------|--------|
| `()`   | Unit type with a single value `()` | 0-bit  |

The unit type `()` has exactly one value, also written `()`. It represents the absence of a meaningful value. In type signatures, `()` appears as the parameter type for zero-parameter functions (e.g., `f : () -> i64`).

### 6.6 Function Types

Function types describe the signature of a function. They are written using the arrow notation in type signatures:

    FunctionType = TypeProduct "->" TypeExpr .

Examples:

    i64 -> i64                   (* function taking i64, returning i64 *)
    i64 * i64 -> i64             (* function taking two i64s, returning i64 *)
    () -> i64                    (* function taking no arguments, returning i64 *)
    i64 -> i64 -> i64            (* curried: equivalent to i64 -> (i64 -> i64) *)

The arrow `->` is right-associative. The product operator `*` (used to separate parameter types) binds more tightly than `->`.

### 6.7 Type Variables

Type variables represent unknown types in polymorphic type signatures. They are written with a leading apostrophe:

    TypeVar = "'" lower_letter { letter | decimal_digit } .

Type variables may have optional trait constraints:

    'a                           (* unconstrained type variable *)
    'a:Numeric                   (* type variable constrained to numeric types *)

Type variables are introduced by `forall` quantification in type signatures or implicitly when a function parameter's type cannot be locally determined.

## 7. Properties of Types and Values

### 7.1 Type Identity

Two types are identical if and only if they have the same type name. For parameterized types (integer, float, index), the parameters must also match.

Specifically:
- `i32` and `i32` are identical.
- `i32` and `u32` are not identical (different signedness).
- `i32` and `i64` are not identical (different width).
- `f32` and `f64` are not identical.
- `isize` and `i64` are not identical, even on 64-bit platforms where they have the same width.

### 7.2 Assignability

A value `v` of type `T` is assignable to a variable of type `U` if and only if `T` and `U` are identical types.

There are no implicit type conversions. If `T` and `U` are different types, an explicit conversion using `as` is required (see [§10.7 Type Conversions](#107-type-conversions)).

### 7.3 Type Compatibility for Operations

Binary operators require both operands to have identical types. The result type of arithmetic operators is the same as the operand type. The result type of comparison operators is always `bool`.

## 8. Blocks and Scope

### 8.1 Blocks

A Polang program is a sequence of top-level statements. The entire program forms the outermost block.

    Program = { Statement } .

Let-expressions and module declarations introduce nested scopes.

### 8.2 Scope Rules

The scope of a declared name is the region of source code where the name is visible:

1. **Top-level declarations**: A name declared at the top level is visible from the point of declaration to the end of the program. A name must be declared before it is used; forward references are not permitted.

2. **Let-expression bindings**: A name bound in a let-expression is visible only within the body of that let-expression.

       let x = 1 in x + 1      (* x is visible only in "x + 1" *)

3. **Function parameters**: Parameters are visible only within the function body.

4. **Module scope**: Names declared inside a module are visible within the module body. Exported names are accessible outside the module via qualified access or import statements.

### 8.3 Shadowing

A name declared in an inner scope shadows any name with the same identifier in an outer scope. The outer name becomes inaccessible within the inner scope.

    x = 10
    let x = 20 in x             (* evaluates to 20; inner x shadows outer x *)

### 8.4 Declaration Order and Forward References

Declarations at the same scope level are processed in order. A name must not be referenced before it is declared, with one exception:

**Forward references via type signatures.** If a type signature for a name appears before its definition, the name is available for reference from the point of the type signature onward. This enables mutual recursion and top-down code organization:

    isEven : i64 -> bool
    isOdd : i64 -> bool

    isEven(n) = if n == 0 then true else isOdd(n - 1)
    isOdd(n) = if n == 0 then false else isEven(n - 1)

A type signature without a corresponding definition in the same scope is ill-formed.

## 9. Declarations

### 9.1 Type Signatures

A type signature declares the type of a name. It must appear on a separate statement immediately before the corresponding definition.

    TypeSignature = identifier ":" TypeExpr .

    TypeExpr     = "forall" TypeVarList "." TypeExpr
                 | TypeProduct "->" TypeExpr
                 | TypeProduct .
    TypeVarList  = TypeVarDecl { "," TypeVarDecl } .
    TypeVarDecl  = typevar [ ":" identifier ] .
    TypeProduct  = TypeAtom { "*" TypeAtom } .
    TypeAtom     = identifier
                 | typevar
                 | "()"
                 | "(" TypeExpr ")" .

Examples:

    x : i64
    add : i64 * i64 -> i64
    identity : forall 'a. 'a -> 'a
    scale : forall 'a:Numeric. 'a * i64 -> 'a

When a type signature is present, the definition's inferred type must match the declared type exactly. If they do not match, the program is ill-formed.

Type signatures for top-level definitions are recommended. The compiler emits a warning when a top-level definition lacks a type signature.

### 9.2 Variable Declarations

A variable declaration binds a name to a value.

    VariableDecl = identifier "=" Expression .

All variables are immutable. Once bound, a variable's value cannot be changed.

A variable must be initialized at the point of declaration. The type of the variable is determined by:
1. An explicit type signature on the preceding line, if present.
2. Otherwise, type inference from the initializer expression.

Examples:

    x = 5                        (* type inferred as i64 *)
    y = 3.14                     (* type inferred as f64 *)

    z : i32
    z = 42                       (* type declared as i32 *)

### 9.3 Function Declarations

A function declaration binds a name to a function value.

    FunctionDecl = identifier "(" [ ParamList ] ")" "=" Expression .
    ParamList    = Param { "," Param } .
    Param        = identifier .

The function body is a single expression. The function's return value is the result of evaluating that expression.

Parameter types and return type are determined by:
1. An explicit type signature on the preceding line, if present.
2. Otherwise, Hindley-Milner type inference (see [§12 Type System](#12-type-system)).

Examples:

    add : i64 * i64 -> i64
    add(x, y) = x + y

    square(n) = n * n            (* types inferred from usage *)

    greet : () -> i64
    greet() = 42                 (* zero-parameter function *)

### 9.4 Module Declarations

A module declaration groups related declarations into a namespace with an explicit export list.

    ModuleDecl = "module" identifier [ "(" IdentList ")" ] ModuleBody "endmodule" .
    IdentList  = identifier { "," identifier } .
    ModuleBody = { VariableDecl | FunctionDecl | TypeSignature | ModuleDecl } .

The export list in parentheses specifies which names are public. Names not in the export list are private to the module. A module without an export list exports no names.

Examples:

    module Math (add, PI)
      PI : f64
      PI = 3.14159

      add : i64 * i64 -> i64
      add(x, y) = x + y

      helper(x) = x * 2         (* private: not in export list *)
    endmodule

Modules may be nested:

    module Outer (Inner)
      module Inner (foo)
        foo : i64 -> i64
        foo(x) = x + 1
      endmodule
    endmodule

### 9.5 Import Statements

Import statements bring module members into the current scope.

    ImportStmt = "import" QualifiedName [ "as" identifier ]
               | "from" QualifiedName "import" ImportItems
               | "from" QualifiedName "import" "*" .
    QualifiedName = identifier { "." identifier } .
    ImportItems   = ImportItem { "," ImportItem } .
    ImportItem    = identifier [ "as" identifier ] .

Forms:

1. `import M` -- makes `M.x`, `M.f()` available via qualified access.
2. `import M as N` -- makes `N.x`, `N.f()` available (alias).
3. `from M import x, f` -- makes `x`, `f` available directly.
4. `from M import x as y` -- makes `y` available (alias for `x`).
5. `from M import *` -- imports all exported names directly.

Only exported names can be imported. Attempting to import a private name is a compile-time error.

## 10. Expressions

### 10.1 Operands

Operands denote the elementary values in an expression.

    Operand = identifier
            | QualifiedName
            | integer_lit
            | float_lit
            | bool_lit
            | unit_lit
            | "(" Expression ")" .

A qualified name accesses a module member:

    Math.PI
    Outer.Inner.foo

Qualified names support arbitrary depth (`a.b`, `a.b.c`, `a.b.c.d`, etc.).

### 10.2 Function Calls

    Call = identifier "(" [ ArgList ] ")"
         | QualifiedName "(" [ ArgList ] ")" .
    ArgList = Expression { "," Expression } .

A function call evaluates each argument expression, then invokes the named function. The number of arguments must match the function's parameter count; otherwise the program is ill-formed.

The order in which argument expressions are evaluated is unspecified. The implementation may evaluate them in any order.

Examples:

    add(1, 2)
    Math.add(1, 2)
    greet()

### 10.3 Unary Operators

    UnaryExpr = UnaryOp Expression .
    UnaryOp   = "-" | "!" .

#### 10.3.1 Arithmetic Negation

| Operator | Operation | Operand Type     | Result Type      |
|----------|-----------|------------------|------------------|
| `-`      | Negation  | Integer or Float | Same as operand  |

Unary `-` computes the arithmetic negation of its operand. For signed integers, negating the minimum value (e.g., `-128` for `i8`) is an overflow and shall cause a runtime error (see [§6.1](#61-integer-types)). For unsigned integers, unary negation is not permitted; the program is ill-formed.

    -x                           (* negation of x *)
    -(a + b)                     (* negation of a sum *)

#### 10.3.2 Logical Not

| Operator | Operation   | Operand Type | Result Type |
|----------|-------------|--------------|-------------|
| `!`      | Logical not | `bool`       | `bool`      |

    !true                        (* false *)
    !false                       (* true *)
    !(a > b)                     (* negation of comparison *)

### 10.4 Binary Operators

    BinaryExpr = Expression BinOp Expression .
    BinOp      = "+" | "-" | "*" | "/" | "%"
               | "&+" | "&-" | "&*"
               | "==" | "!=" | "<" | "<=" | ">" | ">="
               | "&&" | "||" .

#### 10.4.1 Arithmetic Operators

| Operator | Operation      | Operand Types       | Result Type    |
|----------|----------------|---------------------|----------------|
| `+`      | Addition       | Integer or Float    | Same as operands |
| `-`      | Subtraction    | Integer or Float    | Same as operands |
| `*`      | Multiplication | Integer or Float    | Same as operands |
| `/`      | Division       | Integer or Float    | Same as operands |
| `%`      | Remainder      | Integer only        | Same as operands |

Both operands of `+`, `-`, `*`, `/` must have identical types. The result has the same type as the operands.

The `%` operator computes the remainder of integer division (truncated division). The result has the same sign as the dividend. Float operands are not permitted with `%`; the program is ill-formed.

Integer division or remainder by zero shall cause the program to print a runtime error message including the source location and exit with a non-zero status. Floating-point division by zero produces infinity or NaN per IEEE 754.

#### 10.4.2 Wrapping Arithmetic Operators

| Operator | Operation                 | Operand Types    | Result Type      |
|----------|---------------------------|------------------|------------------|
| `&+`     | Wrapping addition         | Integer          | Same as operands |
| `&-`     | Wrapping subtraction      | Integer          | Same as operands |
| `&*`     | Wrapping multiplication   | Integer          | Same as operands |

Wrapping operators perform the same computation as their checked counterparts but silently wrap on overflow instead of causing a runtime error. They are intended for performance-critical code where overflow behavior is intentional (e.g., hash functions, bitwise algorithms).

Wrapping operators are only available for integer types. Using them with floating-point operands is ill-formed.

#### 10.4.3 Comparison Operators

| Operator | Operation              | Operand Types       | Result Type |
|----------|------------------------|---------------------|-------------|
| `==`     | Equal                  | Integer or Float    | `bool`      |
| `!=`     | Not equal              | Integer or Float    | `bool`      |
| `<`      | Less than              | Integer or Float    | `bool`      |
| `<=`     | Less than or equal     | Integer or Float    | `bool`      |
| `>`      | Greater than           | Integer or Float    | `bool`      |
| `>=`     | Greater than or equal  | Integer or Float    | `bool`      |

Both operands must have identical numeric types. Boolean values cannot be compared.

For signed integers, comparisons use signed semantics. For unsigned integers, comparisons use unsigned semantics. For floating-point values, comparisons follow IEEE 754 ordering.

#### 10.4.4 Logical Operators

| Operator | Operation   | Operand Types | Result Type |
|----------|-------------|---------------|-------------|
| `&&`     | Logical and | `bool`        | `bool`      |
| `\|\|`   | Logical or  | `bool`        | `bool`      |

Logical operators use **short-circuit evaluation**:

- `a && b`: If `a` is `false`, the result is `false` and `b` is not evaluated.
- `a || b`: If `a` is `true`, the result is `true` and `b` is not evaluated.

Both operands must have type `bool`.

    a > 0 && a < 100             (* true if a is in range (1, 99) *)
    x == 0 || y == 0             (* true if either is zero *)

### 10.5 Operator Precedence

Operators are listed from highest to lowest precedence:

| Precedence | Operators                           | Associativity   |
|------------|-------------------------------------|-----------------|
| 9          | `.` (member access)                 | Left            |
| 8          | Unary `-`, `!`                      | Right (prefix)  |
| 7          | `as` (type conversion)              | Left            |
| 6          | `*`, `/`, `%`, `&*`                 | Left            |
| 5          | `+`, `-`, `&+`, `&-`               | Left            |
| 4          | `==`, `!=`, `<`, `<=`, `>`, `>=`    | Non-associative |
| 3          | `&&`                                | Left            |
| 2          | `\|\|`                              | Left            |
| 1          | `if`/`then`/`else`, `let`/`in`      | Right           |

Note: The `=` symbol is a definition separator, not an expression operator, and does not appear in the precedence table.

Parentheses override precedence:

    (1 + 2) * 3                  (* 9, not 7 *)
    a + b as i32                 (* means: a + (b as i32) *)
    !a && b                      (* means: (!a) && b *)
    a > 0 && a < 10 || b == 0   (* means: ((a > 0) && (a < 10)) || (b == 0) *)

Comparison operators are non-associative; chaining comparisons like `a < b < c` is a syntax error.

### 10.6 If Expressions

    IfExpr = "if" Expression "then" Expression "else" Expression .

An if-expression evaluates the condition, then evaluates exactly one of the two branches:
- If the condition is `true`, the `then` branch is evaluated.
- If the condition is `false`, the `else` branch is evaluated.

The condition must have type `bool`. Both branches must have identical types. The type of the if-expression is the type of its branches.

Both branches are required. There is no if-without-else form.

If-expressions can be nested:

    if a > 0 then 1 else if a < 0 then -1 else 0

The `-1` in the example above is parsed as unary negation applied to `1`.

### 10.7 Type Conversions

    CastExpr = Expression "as" identifier .

The `as` operator performs an explicit type conversion. The target type must be a numeric type name (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `isize`, `usize`). Only numeric-to-numeric conversions are permitted. Boolean conversions are not allowed. If the target type is not a valid numeric type, the program is ill-formed.

#### 10.7.1 Allowed Conversions

| From              | To                | Allowed |
|-------------------|-------------------|---------|
| Any integer       | Any integer       | Yes     |
| Any integer       | Any float         | Yes     |
| Any integer       | Any index         | Yes     |
| Any float         | Any integer       | Yes     |
| Any float         | Any float         | Yes     |
| Any float         | Any index         | Yes     |
| Any index         | Any integer       | Yes     |
| Any index         | Any float         | Yes     |
| Any index         | Any index         | Yes     |
| `bool`            | Any type          | No      |
| Any type          | `bool`            | No      |

If a conversion is not allowed, the program is ill-formed.

#### 10.7.2 Integer Widening

Converting to a larger integer type preserves the value exactly:

- **Signed to larger signed**: sign-extension.
- **Unsigned to larger unsigned**: zero-extension.
- **Unsigned to larger signed**: zero-extension.
- **Signed to larger unsigned**: sign-extension, then reinterpretation as unsigned.

    (-1 as i8) as i64             (* -1: sign-extended *)
    (255 as u8) as u64            (* 255: zero-extended *)
    (-1 as i8) as u64             (* 18446744073709551615: reinterpreted *)

#### 10.7.3 Integer Narrowing

Converting to a smaller integer type truncates, keeping only the low-order bits (wrap-around):

    (256 as i32) as i8            (* 0: 256 mod 256 *)
    (257 as i32) as i8            (* 1: 257 mod 256 *)

#### 10.7.4 Sign Reinterpretation

Converting between signed and unsigned types of the same width reinterprets the bit pattern without changing it:

    (-1 as i32) as u32            (* 4294967295 *)
    (4294967295 as u32) as i32    (* -1 *)

#### 10.7.5 Float Widening

Converting `f32` to `f64` is exact with no precision loss.

#### 10.7.6 Float Narrowing

Converting `f64` to `f32` rounds to the nearest representable value using IEEE 754 round-to-nearest, ties-to-even. Very large values may overflow to positive or negative infinity.

#### 10.7.7 Integer to Float

The integer value is converted to the nearest representable floating-point value. Large integers (beyond the mantissa precision of the target float) may lose precision.

#### 10.7.8 Float to Integer

Float-to-integer conversion uses **saturating truncation toward zero**:

1. The fractional part is discarded (truncation toward zero).
2. If the result exceeds the target integer range, it is clamped to the nearest boundary value.
3. NaN converts to 0.

Examples:

    3.7 as i32                    (* 3: truncated *)
    (-3.7 as f64) as i32          (* -3: truncated toward zero *)
    1000.0 as i8                  (* 127: saturated at i8 max *)
    (-1000.0 as f64) as i8        (* -128: saturated at i8 min *)

#### 10.7.9 Index Conversions

- **Integer to index**: Uses signed or unsigned index cast depending on the target type (`isize` or `usize`).
- **Index to integer**: Uses the corresponding signed or unsigned cast.
- **Float to index**: Two-step conversion: float to `i64` (saturating truncation), then `i64` to index.
- **Index to float**: Two-step conversion: index to `i64`, then `i64` to float.

### 10.8 Let Expressions

    LetExpr     = "let" LetBindings "in" Expression .
    LetBindings = LetBinding { "and" LetBinding } .
    LetBinding  = identifier "=" Expression
                | identifier ":" identifier "=" Expression
                | identifier "(" [ TypedParamList ] ")" [ ":" identifier ] "=" Expression .
    TypedParamList = TypedParam { "," TypedParam } .
    TypedParam     = identifier [ ":" identifier ] .

A let-expression introduces one or more local bindings that are visible only within the body expression. The entire let-expression evaluates to the value of the body.

Bindings can be variable bindings or function bindings, and can be mixed:

    let x = 1 in x + 1
    let x = 1 and y = 2 in x + y
    let f(n: i64): i64 = n + 1 in f(5)
    let x = 10 and double(y: i64): i64 = y * 2 in double(x)

Multiple bindings separated by `and` are evaluated independently. Bindings within the same `let` group cannot reference each other. The order in which binding expressions are evaluated is unspecified.

## 11. Statements

A Polang program consists of a sequence of top-level statements.

    Statement = TypeSignature
              | VariableDecl
              | FunctionDecl
              | ModuleDecl
              | ImportStmt
              | ExpressionStmt .
    ExpressionStmt = Expression .

### 11.1 Type Signature Statements

A type signature statement (see [§9.1](#91-type-signatures)) declares the type of the next definition.

### 11.2 Definition Statements

A variable or function definition (see [§9.2](#92-variable-declarations) and [§9.3](#93-function-declarations)) binds a name to a value.

### 11.3 Module Declarations

A module declaration (see [§9.4](#94-module-declarations)) introduces a namespace.

### 11.4 Import Statements

An import statement (see [§9.5](#95-import-statements)) brings module members into scope.

### 11.5 Expression Statements

An expression appearing as a top-level statement is evaluated for its value. In the REPL, the value of the last expression statement is printed. In the compiler, the last expression in the program determines the return value of the entry function.

## 12. Type System

### 12.1 Overview

Polang uses a Hindley-Milner style type system with:

- **Static typing**: All types are determined at compile time.
- **Type inference**: Types can be inferred from context.
- **Polymorphic functions**: Functions can work with multiple types via type variables.
- **Explicit conversions only**: No implicit type coercions.
- **Monomorphization**: Polymorphic functions are specialized at compile time.

### 12.2 Type Inference

Type inference determines the types of expressions, variables, and function parameters when explicit type annotations are not provided.

#### 12.2.1 Local Type Inference

When a function parameter's type is not declared, the type checker infers it from how the parameter is used in the function body:

- If combined with an integer literal via arithmetic (`x + 1`, `x * 2`), the parameter is inferred as `i64`.
- If combined with a floating-point literal (`x + 1.0`, `x / 2.0`), the parameter is inferred as `f64`.
- If used as a condition in an if-expression (`if x then ...`), the parameter is inferred as `bool`.
- If used with another parameter or variable of known type via a binary operation, the parameter inherits that type.

#### 12.2.2 Polymorphic Type Inference

When a parameter's type cannot be determined from local usage, it becomes a type variable. The concrete type is determined at each call site:

    identity(x) = x
    identity(42)                 (* x resolved to i64 at this call *)
    identity(true)               (* x resolved to bool at this call *)

Each distinct call-site type combination produces a specialized function via monomorphization.

### 12.3 Unification Algorithm

The type checker uses the Hindley-Milner unification algorithm:

1. **Constraint collection**: Type constraints are gathered from expressions. Arithmetic operations require operands and result to share the same type. Function calls require argument types to match parameter types.

2. **Unification**: Constraints are solved using unification. Type variables can be bound to concrete types or other type variables. An occurs check prevents infinite types.

3. **Substitution**: The resulting substitution is applied to resolve all type variables to concrete types.

### 12.4 Literal Type Inference

Numeric literals adapt to their declared type context:

- If a variable has an explicit type signature, the literal takes that type, provided the value fits within the type's range.
- If no type context is available, integer literals default to `i64` and floating-point literals default to `f64`.

If the literal value does not fit in the target type, the program is ill-formed (see [§5.1](#51-numeric-constants)).

### 12.5 If-Expression Typing

For an if-expression `if C then A else B`:

1. `C` must have type `bool`.
2. `A` and `B` must have identical types.
3. The type of the entire expression is the type of `A` (and `B`).

If these conditions are not met, the program is ill-formed.

### 12.6 Monomorphization

Polymorphic functions are compiled by creating specialized copies for each unique set of concrete types observed at call sites.

For a polymorphic function `f` with type variable `'a`:
- Each call `f(v)` where `v` has a concrete type `T` creates (if not already present) a specialized version `f$T`.
- The call is replaced with a call to the specialized version.
- Specialized function names use the mangling convention `name$type1$type2`.

A polymorphic function that is never called produces no specialized copies and is erased during compilation.

## 13. Modules

### 13.1 Module Structure

A module is a named collection of declarations with an explicit export list that controls visibility (see [§9.4](#94-module-declarations)).

### 13.2 Qualified Access

Module members are accessed using dot notation:

    Math.PI
    Math.add(1, 2)
    Outer.Inner.foo(5)

Only exported members can be accessed from outside the module. Accessing a private member is a compile-time error.

### 13.3 Import Resolution

Import statements (see [§9.5](#95-import-statements)) resolve names as follows:

1. `import M` -- creates a module alias `M` in the current scope.
2. `import M as N` -- creates a module alias `N` in the current scope.
3. `from M import x` -- binds `x` in the current scope to `M.x`.
4. `from M import x as y` -- binds `y` in the current scope to `M.x`.
5. `from M import *` -- binds all exported names of `M` in the current scope.

Name conflicts from wildcard imports or multiple imports of the same name are a compile-time error.

## 14. Built-in Operations

### 14.1 Printing

In the REPL, the value of the last expression is automatically printed to stdout with its type:

    > 42
    42 : i64

    > 3.14
    3.14 : f64

The format is `value : type` followed by a newline.

In compiler mode (file execution via the REPL), the last top-level expression value is printed in the same format.

### 14.2 Entry Function

The compiler wraps all top-level code in an entry function called `__polang_entry`. This function returns the value of the last expression in the program. If the program has no expression statements, the entry function returns a default value.

## 15. Program Execution

### 15.1 Program Structure

A valid Polang program is a sequence of statements as defined in [§11](#11-statements). An empty program (or a program containing only comments) is valid.

### 15.2 Evaluation Order

Top-level statements are evaluated in order from first to last. Within an expression, evaluation order follows standard rules:
- Function arguments are evaluated before the function is called.
- Binary operator operands are evaluated (left operand first, then right operand) before the operation.
- In `let` expressions with multiple bindings, each binding expression is evaluated independently.

### 15.3 Compilation Modes

Polang supports two execution modes:

1. **Compiler mode**: Source code is compiled to LLVM IR. The `__polang_entry` function serves as the program entry point.

2. **REPL mode**: Statements are compiled and executed incrementally. Variables and functions persist across evaluations. Each evaluation creates a new entry function (`__polang_eval_0`, `__polang_eval_1`, etc.).

### 15.4 Closures

Functions can capture variables from their enclosing scope:

    x = 10
    f() = x + 1

Captured variables are passed by value as implicit extra parameters at the call site. The captured value is the value at the time of the function call.

## 16. Compile-time Errors

The following conditions make a program ill-formed. The compiler shall report an error with source location information (line and column).

### 16.1 Syntax Errors

- Unexpected token in input.
- Unterminated comment.
- Invalid character in source.
- Invalid left-hand side of a definition.
- Missing `else` branch in an if-expression.

### 16.2 Scope Errors

- Reference to an undeclared variable.
- Reference to a private module member from outside the module.

### 16.3 Type Errors

- Type mismatch in binary operation (operands have different types).
- Type mismatch in if-expression (branches have different types).
- Non-boolean condition in if-expression.
- Function call with wrong number of arguments.
- Function call with argument type mismatch (when types are known).
- Type signature does not match inferred type.
- Invalid type conversion (e.g., `bool` to integer).
- Literal value does not fit in target type.

### 16.4 Error Message Format

Type checker errors use the format:

    Type error: <message> at line <line>, column <column>

Syntax errors use the format:

    ERROR: <message> at line <line>, column <column>

## 17. System Considerations

### 17.1 Size and Alignment

The sizes of all types are specified in [§6](#6-types). Integer and floating-point types have the exact widths listed. The `bool` type occupies 1 bit logically but may be stored in a larger unit at the implementation's discretion (following LLVM's `i1` convention).

### 17.2 Platform Dependence

The following aspects are platform-dependent:

- The width of `isize` and `usize` (equal to the platform pointer width).
- Integer overflow behavior for signed types follows two's complement (guaranteed by LLVM).

### 17.3 Compilation Pipeline

The implementation compiles Polang source through the following stages:

1. **Lexing**: Source text to tokens (Flex).
2. **Parsing**: Tokens to AST (Bison).
3. **Type checking**: AST-level Hindley-Milner type inference and validation.
4. **MLIR generation**: AST to Polang dialect MLIR.
5. **Monomorphization**: Generic function specialization.
6. **Lowering**: Polang dialect to standard MLIR dialects (arith, func, scf).
7. **LLVM lowering**: Standard dialects to LLVM dialect to LLVM IR.

### 17.4 REPL Considerations

The REPL maintains persistent state across evaluations:

- Variables declared in one evaluation are accessible in subsequent evaluations.
- Functions declared in one evaluation can be called in subsequent evaluations.
- The type checker maintains a snapshot/rollback mechanism: if type checking fails for new input, the state rolls back to before that input.
- Each evaluation compiles to a separate JIT module (JITDylib). Previously declared globals are accessed via external symbol declarations.
