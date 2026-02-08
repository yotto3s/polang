# Polang Type System

This document describes the Polang type system, including type inference and polymorphic types.

## Table of Contents

- [Overview](#overview)
- [Primitive Types](#primitive-types)
- [Type Conversions](#type-conversions)
  - [Conversion Syntax](#conversion-syntax)
  - [Allowed Conversions](#allowed-conversions)
  - [Conversion Semantics](#conversion-semantics)
  - [Literal Type Inference](#literal-type-inference)
  - [Boolean Conversions](#boolean-conversions)
- [Type Inference](#type-inference)
  - [Local Type Inference](#local-type-inference)
  - [If-Expression Type Inference](#if-expression-type-inference)
  - [Polymorphic Type Inference](#polymorphic-type-inference)
- [Type Variables](#type-variables)
- [Unification Algorithm](#unification-algorithm)
- [Implementation Details](#implementation-details)
- [Examples](#examples)
- [Monomorphization](#monomorphization)
- [Error Handling](#error-handling)
- [Known Limitations](#known-limitations)

## Overview

Polang uses a Hindley-Milner style type system with:

- **Static typing**: All types are determined at compile time
- **Type inference**: Types can be inferred from context, reducing annotation burden
- **Polymorphic functions**: Functions can work with multiple types via type variables
- **Explicit conversions only**: No implicit type coercions; use `as` for numeric conversions

The type system is implemented in two phases:

1. **Parser-level type checking**: Validates types, detects errors early, performs Hindley-Milner type inference, and performs capture analysis for closures
2. **MLIR-level passes**: Resolves type parameters in generic functions via monomorphization

## Primitive Types

Polang supports a variety of numeric types with explicit width and signedness, plus a boolean type:

### Integer Types

| Type | Description | Size | MLIR Type | LLVM Type |
|------|-------------|------|-----------|-----------|
| `i8` | Signed 8-bit integer | 8-bit | `!polang.integer<8, signed>` | `i8` |
| `i16` | Signed 16-bit integer | 16-bit | `!polang.integer<16, signed>` | `i16` |
| `i32` | Signed 32-bit integer | 32-bit | `!polang.integer<32, signed>` | `i32` |
| `i64` | Signed 64-bit integer | 64-bit | `!polang.integer<64, signed>` | `i64` |
| `u8` | Unsigned 8-bit integer | 8-bit | `!polang.integer<8, unsigned>` | `i8` |
| `u16` | Unsigned 16-bit integer | 16-bit | `!polang.integer<16, unsigned>` | `i16` |
| `u32` | Unsigned 32-bit integer | 32-bit | `!polang.integer<32, unsigned>` | `i32` |
| `u64` | Unsigned 64-bit integer | 64-bit | `!polang.integer<64, unsigned>` | `i64` |

### Floating-Point Types

| Type | Description | Size | MLIR Type | LLVM Type |
|------|-------------|------|-----------|-----------|
| `f32` | Single-precision float | 32-bit | `!polang.float<32>` | `f32` |
| `f64` | Double-precision float | 64-bit | `!polang.float<64>` | `f64` |

### Index Types

| Type | Description | Size | MLIR Type | LLVM Type |
|------|-------------|------|-----------|-----------|
| `isize` | Signed index (pointer width) | Platform-dependent | `!polang.index<signed>` | `index` |
| `usize` | Unsigned index (pointer width) | Platform-dependent | `!polang.index<unsigned>` | `index` |

Index types map to the platform-native pointer width. `usize` is intended for array indexing (Phase 2). Both types support explicit casting with `as` to/from integer and float types.

### Boolean Type

| Type | Description | Size | MLIR Type | LLVM Type |
|------|-------------|------|-----------|-----------|
| `bool` | Boolean | 1-bit | `!polang.bool` | `i1` |

### Type Constants

Type names are defined as compile-time constants in `parser/include/parser/polang_types.hpp`:

```cpp
struct TypeNames {
  // Signed integers
  static constexpr const char* I8 = "i8";
  static constexpr const char* I16 = "i16";
  static constexpr const char* I32 = "i32";
  static constexpr const char* I64 = "i64";

  // Unsigned integers
  static constexpr const char* U8 = "u8";
  static constexpr const char* U16 = "u16";
  static constexpr const char* U32 = "u32";
  static constexpr const char* U64 = "u64";

  // Floating-point
  static constexpr const char* F32 = "f32";
  static constexpr const char* F64 = "f64";

  // Index types
  static constexpr const char* ISIZE = "isize";
  static constexpr const char* USIZE = "usize";

  // Other types
  static constexpr const char* BOOL = "bool";
  static constexpr const char* FUNCTION = "function";
  static constexpr const char* TYPEVAR = "typevar";
  static constexpr const char* UNKNOWN = "unknown";
};
```

### Literal Type Defaults

When type inference encounters a numeric literal without explicit type context, it defaults to:

- **Integer literals** → `i64` (signed 64-bit)
- **Float literals** → `f64` (64-bit double precision)

This means `42` has type `i64` and `3.14` has type `f64` by default.

## Type Conversions

Polang supports explicit type conversions between numeric types using the `as` keyword. All conversions must be explicit; there are no implicit type coercions.

### Conversion Syntax

The `as` keyword converts a value from one numeric type to another:

```polang
expression as TargetType
```

**Examples:**

```polang
let a: i32 = 1000
let b: i64 = a as i64       ; widen i32 to i64
let c: i8 = a as i8         ; narrow i32 to i8 (truncates)
let d: f64 = a as f64       ; convert integer to float
let e: i32 = 3.14 as i32    ; convert float to integer (truncates toward zero)
```

### Operator Precedence

The `as` operator has higher precedence than arithmetic operators but lower precedence than unary operators:

| Precedence | Operators |
|------------|-----------|
| Highest | Parentheses, function calls |
| | Unary `-`, `!` |
| | **`as` (type conversion)** |
| | `*`, `/`, `%` |
| | `+`, `-` |
| | `<`, `>`, `<=`, `>=`, `==`, `!=` |
| | `&&` |
| Lowest | `\|\|` |

**Precedence examples:**

```polang
a + b as i32        ; means: a + (b as i32)
-x as i32           ; means: (-x) as i32
a * b as f64 + c    ; means: (a * (b as f64)) + c
(a + b) as i32      ; parentheses override precedence
```

### Allowed Conversions

Only numeric-to-numeric conversions are permitted:

| From Type | To Type | Allowed |
|-----------|---------|---------|
| Any integer (`i8`-`i64`, `u8`-`u64`) | Any integer | ✓ |
| Any integer | Any float (`f32`, `f64`) | ✓ |
| Any integer | Any index (`isize`, `usize`) | ✓ |
| Any float | Any integer | ✓ |
| Any float | Any float | ✓ |
| Any float | Any index | ✓ |
| Any index | Any integer | ✓ |
| Any index | Any float | ✓ |
| Any index | Any index | ✓ |
| `bool` | Any type | ✗ |
| Any type | `bool` | ✗ |

**Type error examples:**

```polang
let x: i32 = true as i32    ; Error: cannot convert bool to i32
let b: bool = 1 as bool     ; Error: cannot convert i32 to bool
```

### Conversion Semantics

#### Integer Widening

Converting to a larger integer type preserves the value exactly:

| Conversion | Operation | Example |
|------------|-----------|---------|
| Signed → Larger signed | Sign-extend | `(-1i8) as i64` → `-1i64` |
| Unsigned → Larger unsigned | Zero-extend | `255u8 as u64` → `255u64` |
| Unsigned → Larger signed | Zero-extend | `255u8 as i64` → `255i64` |
| Signed → Larger unsigned | Sign-extend then reinterpret | `(-1i8) as u64` → `18446744073709551615u64` |

#### Integer Narrowing

Converting to a smaller integer type truncates the value, keeping only the low-order bits (wrap-around behavior):

```polang
let a: i32 = 256
let b: i8 = a as i8          ; b = 0 (256 mod 256)

let c: i32 = 257
let d: i8 = c as i8          ; d = 1 (257 mod 256)

let e: i32 = -1
let f: u8 = e as u8          ; f = 255 (bit pattern preserved)
```

#### Sign Reinterpretation

Converting between signed and unsigned types of the same width reinterprets the bit pattern:

```polang
let a: i32 = -1
let b: u32 = a as u32        ; b = 4294967295 (same bits, different interpretation)

let c: u32 = 4294967295
let d: i32 = c as i32        ; d = -1
```

#### Float Widening

Converting `f32` to `f64` is always exact with no precision loss:

```polang
let a: f32 = 3.14
let b: f64 = a as f64        ; exact conversion
```

#### Float Narrowing

Converting `f64` to `f32` rounds to the nearest representable value (IEEE 754 round-to-nearest, ties-to-even):

```polang
let a: f64 = 3.141592653589793
let b: f32 = a as f32        ; rounded to f32 precision
```

Very large `f64` values may overflow to `±infinity` when converted to `f32`.

#### Integer to Float

Integer values are converted to the nearest representable floating-point value:

```polang
let a: i64 = 42
let b: f64 = a as f64        ; b = 42.0

let c: i64 = 9007199254740993   ; larger than f64 can represent exactly
let d: f64 = c as f64           ; may lose precision
```

Note: `f64` has 53 bits of mantissa precision, so `i64` values larger than 2^53 may not be exactly representable.

#### Float to Integer

Float-to-integer conversion uses **saturating truncation toward zero**:

1. **Truncation**: The fractional part is discarded, rounding toward zero
2. **Saturation**: Values outside the target range are clamped to the minimum or maximum
3. **NaN handling**: `NaN` converts to `0`

```polang
; Truncation toward zero
let a: i32 = 3.7 as i32      ; a = 3
let b: i32 = -3.7 as i32     ; b = -3

; Saturation at bounds
let c: i8 = 1000.0 as i8     ; c = 127 (i8 max)
let d: i8 = -1000.0 as i8    ; d = -128 (i8 min)
let e: u8 = -1.0 as u8       ; e = 0 (u8 min, since negative)
let f: u8 = 1000.0 as u8     ; f = 255 (u8 max)

; Infinity handling
let g: i32 = (1.0 / 0.0) as i32   ; g = 2147483647 (i32 max)
let h: i32 = (-1.0 / 0.0) as i32  ; h = -2147483648 (i32 min)

; NaN handling
let i: i32 = (0.0 / 0.0) as i32   ; i = 0
```

#### Integer to Index

Converting an integer to `isize` or `usize` uses `arith.index_cast` (signed) or `arith.index_castui` (unsigned):

```polang
def a: i64 = 42
def b: isize = a as isize       ; signed index cast
def c: usize = a as usize       ; unsigned index cast
```

#### Index to Integer

Converting `isize` or `usize` to an integer type uses the corresponding signed or unsigned index cast:

```polang
def a: isize = 42 as isize
def b: i64 = a as i64           ; signed index cast to i64
```

#### Float to/from Index

Float-to-index and index-to-float conversions use a two-step process via `i64`:

```polang
def a: f64 = 3.14
def b: isize = a as isize       ; float → i64 (saturating truncation) → index

def c: isize = 42 as isize
def d: f64 = c as f64           ; index → i64 → float
```

### Literal Type Inference

Numeric literals adapt to their declared type context when the value fits:

```polang
let a: i8 = 42               ; OK: 42 fits in i8, literal is i8
let b: i16 = 1000            ; OK: 1000 fits in i16, literal is i16
let c: f32 = 3.14            ; OK: literal becomes f32
let d: u8 = 200              ; OK: 200 fits in u8
```

**Compile-time error** if the literal value doesn't fit in the target type:

```polang
let x: i8 = 1000             ; Error: 1000 doesn't fit in i8 (-128 to 127)
let y: u8 = 256              ; Error: 256 doesn't fit in u8 (0 to 255)
let z: u8 = -1               ; Error: -1 doesn't fit in u8 (unsigned)
```

When no type context is available, literals default to `i64` (integers) or `f64` (floats):

```polang
let a = 42                   ; a: i64
let b = 3.14                 ; b: f64
```

### Boolean Conversions

Boolean values cannot be converted to or from numeric types using `as`. Use explicit expressions instead:

```polang
; Instead of: x as bool (not allowed)
let is_nonzero: bool = x != 0

; Instead of: b as i32 (not allowed)
let int_value: i32 = if b then 1 else 0
```

This design ensures that boolean semantics are always explicit in the code.

## Type Inference

### Parser-Level Type Checking

The parser's type checker (`parser/src/type_checker.cpp`) handles **error detection**, **type inference** via Hindley-Milner unification, and **capture analysis**. Untyped polymorphic parameters are marked as `typevar` and represented as `!polang.type_param` in MLIR for monomorphization.

**Type Checker Responsibilities:**

| Responsibility | Description |
|----------------|-------------|
| Error detection | Undefined variables, arity mismatch |
| Type validation | Validates explicit type annotations match usage |
| Capture analysis | Identifies free variables for closures via `FreeVariableCollector` |
| Type variable setup | Marks untyped polymorphic parameters as `typevar` for MLIR monomorphization |

**Example:**

```polang
let double_it(x) = x * 2    ; x marked as typevar, MLIR infers int
let half(x) = x / 2.0       ; x marked as typevar, MLIR infers double
let identity(x) = x         ; x marked as typevar, MLIR infers from call site
```

**Implementation:**

The type checker uses the `FreeVariableCollector` class for capture analysis:

```cpp
class FreeVariableCollector : public Visitor {
  std::set<std::string> localNames;
  std::set<std::string> referencedNonLocals;

  void visit(const NIdentifier& node) override {
    // If not locally defined, it's a free variable (capture)
    if (localNames.find(node.name) == localNames.end()) {
      referencedNonLocals.insert(node.name);
    }
  }
};
```

### If-Expression Type Inference

If-expressions in Polang are typed based on their branches:

1. **Condition**: Must be `bool`
2. **Branches**: Both `then` and `else` branches must have the same type
3. **Result**: The if-expression's type is the type of both branches

**Example:**

```polang
let a = if true then 1 else 2        ; a is int (branches are int)
let b = if false then 1.0 else 2.0   ; b is double (branches are double)
let c = if true then true else false ; c is bool (branches are bool)
```

**Type Error:**

```polang
let x = if true then 1 else 2.0  ; Error: branches have different types
```

The type checker validates branch type consistency in `TypeChecker::visit(const NIfExpression&)`.

### MLIR Type Inference

All untyped parameters are marked as **type variables** by the parser. Type inference happens primarily at the AST level in the TypeChecker. For polymorphic functions, the MLIR representation uses `!polang.type_param<"name">` type parameters in `polang.generic_func` operations, which are resolved during monomorphization.

**Example:**

```polang
let identity(x) = x     ; x marked as typevar
identity(42)            ; call site constrains x to int
```

**Pipeline:**

```
Source: let identity(x) = x
           ↓
Parser: x marked as TYPEVAR (type checker)
           ↓
MLIRGen: generic_func @identity<a>(%arg0: !polang.type_param<"a">) -> !polang.type_param<"a">
           ↓
Monomorphization: Creates identity$i64 specialized version from instantiate call sites
           ↓
Result: fn identity$i64(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed>
```

## Type Variables

Type variables represent unknown types that will be resolved during type inference.

### Representation

In the parser, type variables are represented as the string `"typevar"`:

```cpp
// When local inference fails, assign a type variable
mutable_arg.type = new NIdentifier(TypeNames::TYPEVAR);
```

In MLIR, type parameters are named types used in generic function templates:

```mlir
!polang.type_param<"a">    ; Named type parameter "a"
!polang.type_param<"b">    ; Named type parameter "b"
```

Type parameters use ML-style naming convention ('a, 'b, etc.) and appear in `polang.generic_func` signatures. They are bound to concrete types by `polang.instantiate` operations at call sites.

### Definition

Type parameters are defined in `mlir/include/polang/Dialect/PolangTypes.td`:

```tablegen
def Polang_TypeParamType : Polang_Type<"TypeParam", "type_param"> {
  let summary = "Named type parameter for generic functions";
  let description = [{
    Represents a named type parameter in a generic function template.
    Uses ML-style naming convention ('a, 'b, etc.).
    Example: !polang.type_param<"a">
  }];
  let parameters = (ins StringRefParameter<"">:$name);
  let assemblyFormat = "`<` $name `>`";
}
```

### Generation

The `MLIRGenVisitor` generates type parameters for untyped function parameters in generic functions. Type parameters are named based on parameter position (e.g., "a", "b", "c").

## Unification Algorithm

Type inference uses the standard Hindley-Milner unification algorithm at the AST level in the TypeChecker (`parser/src/type_checker.cpp`). The algorithm:

1. **Collects constraints** from expressions:
   - Arithmetic operations: operands and result must have same type
   - Function calls: argument types must match parameter types
   - Return statements: return value must match function return type

2. **Unifies types** using the standard algorithm:
   - Type variables can be bound to concrete types or other type variables
   - Occurs check prevents infinite types
   - Constraints are solved to produce a substitution mapping

3. **Applies substitution** to resolve all type variables to concrete types

At the MLIR level, type parameters in `polang.generic_func` operations are resolved during monomorphization, where specialized function copies are created for each unique set of concrete type arguments from `polang.instantiate` call sites.

## Implementation Details

### File Structure

```
parser/
├── include/parser/
│   ├── polang_types.hpp     # Type constants (INT, DOUBLE, BOOL, TYPEVAR)
│   ├── type_checker.hpp     # Type checker interface
│   └── type_inference.hpp   # Unification, substitution, trait constraints,
│                            # and function signature types (MonoSignature,
│                            # PolymorphicSignature, FunctionSignature variant)
└── src/
    └── type_checker.cpp     # Type inference, error detection, capture analysis
                             # Contains: TypeChecker, FreeVariableCollector

mlir/
├── include/polang/
│   ├── Dialect/
│   │   ├── PolangTypes.td   # TypeParamType definition
│   │   └── PolangTypes.h    # Generated type classes
│   └── Transforms/
│       └── Passes.h         # Monomorphization pass declarations
└── lib/
    ├── Dialect/
    │   ├── PolangTypes.cpp          # Type implementations
    │   └── PolangTypeInference.cpp  # MLIR-level type inference pass
    └── Transforms/
        └── Monomorphization.cpp     # Function specialization for call sites
```

### Function Signature Types

The type checker tracks function signatures using a `std::variant`-based type defined in `type_inference.hpp`:

- **`MonoSignature`**: Concrete parameter types and return type (e.g., `(i64, i64) -> i64`)
- **`PolymorphicSignature`**: Type-parameterised signature with trait bounds (e.g., `forall 'a:Numeric. ('a, 'a) -> 'a`)
- **`FunctionSignature`**: `std::variant<MonoSignature, PolymorphicSignature>` — the type checker stores one entry per function in `functionSignatures`

This variant design makes the monomorphic/polymorphic distinction explicit at the type level, preventing bugs from accidentally treating one as the other.

### Type Checking Flow

```
1. Parser builds AST
   └── Type annotations stored in NVariableDeclaration::type

2. Type checker runs (type_checker.cpp)
   ├── Validates explicit type annotations
   ├── Detects errors (undefined vars, arity)
   ├── Performs Hindley-Milner type inference (unification)
   ├── Stores function signatures in functionSignatures map
   ├── Performs capture analysis (FreeVariableCollector)
   └── Marks untyped polymorphic parameters as TYPEVAR

3. MLIRGen generates MLIR (MLIRGen.cpp)
   ├── Converts types to MLIR types (!polang.integer, etc.)
   ├── Generates polang.generic_func with !polang.type_param<"name"> for polymorphic functions
   └── Generates polang.instantiate at call sites with concrete type bindings

4. Type inference pass runs (PolangTypeInference.cpp)
   └── Resolves residual type parameters in generic functions

5. Monomorphization pass runs (Monomorphization.cpp)
   ├── Creates specialized function copies for each unique type combination
   └── Replaces polang.instantiate with polang.call to specialized functions

6. Lowering proceeds with fully-typed MLIR
```

### Verifier Compatibility

Operation verifiers allow type parameters during intermediate stages (before monomorphization resolves them):

```cpp
// In PolangOps.cpp
namespace {
bool typesAreCompatible(Type t1, Type t2) {
  if (t1 == t2) return true;
  // Allow type parameters - they will be resolved during monomorphization
  if (isa<TypeParamType>(t1) || isa<TypeParamType>(t2)) return true;
  return false;
}
}

LogicalResult ReturnOp::verify() {
  // ...
  if (!typesAreCompatible(getValue().getType(), expectedType))
    return emitOpError("type mismatch");
  return success();
}
```

## Examples

### Example 1: Simple Type Inference

```polang
let add(x: i64, y) = x + y
add(1, 2)
```

**Inference:**
- `x` is explicitly `i64`
- `y` is marked as `typevar` by the parser, inferred as `i64` from `x + y`
- Return type inferred as `i64` (result of `+`)
- Monomorphization creates `add$i64_i64`

### Example 2: Polymorphic Identity

```polang
let identity(x) = x
identity(42)
```

**Inference:**
- `x` has no local constraints → assigned as type parameter `"a"`
- Return type depends on `x` → same type parameter `"a"`
- Call `identity(42)` provides concrete type `i64` via `polang.instantiate`
- Monomorphization creates specialized `identity$i64: (i64) -> i64`

### Example 3: Multiple Parameters

```polang
let first(x, y) = x
first(1, 2.0)
```

**Inference:**
- `x` and `y` both get type variables
- Call site: `x` constrained to `i64` (integer literal), `y` constrained to `f64` (float literal)
- Final type: `first$i64_f64: (i64, f64) -> i64`

### Example 4: Inference from Operations

```polang
let is_positive(x) = x > 0
is_positive(5)
```

**Inference:**
- `x` marked as `typevar` by parser, inferred as integer from `x > 0`
- At call site, `5` is an integer literal → resolves to `i64`
- Return type is `bool` (result of comparison)
- Final type: `is_positive$i64: (i64) -> bool`

### Example 5: Float Type Inference

```polang
let add(x, y) = x + y
add(1.5, 2.0)
```

**Inference:**
- `x` and `y` get type variables with `Float` kind (from float literals at call site)
- Float-kind type variables default to `f64`
- Final type: `add$f64_f64: (f64, f64) -> f64`

## Monomorphization

Polymorphic functions are specialized (monomorphized) for each unique set of argument types at call sites.

### How It Works

1. **Identification**: Functions with type variables in their signature are marked as polymorphic
2. **Call Site Analysis**: Each call to a polymorphic function records the concrete argument types
3. **Specialization**: A new function is created for each unique type signature
4. **Name Mangling**: Specialized functions use a mangled name (e.g., `identity$int`)
5. **Call Update**: Calls are updated to reference the specialized version

### Example

```polang
let identity(x) = x
identity(42)     ; Creates identity$i64
identity(true)   ; Creates identity$bool
```

**Generated MLIR:**

```mlir
polang.generic_func @identity<a>(%arg0: !polang.type_param<"a">) -> !polang.type_param<"a"> { ... }

polang.func @identity$i64(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> { ... }

polang.func @identity$bool(%arg0: !polang.bool) -> !polang.bool { ... }
```

### Uncalled Polymorphic Functions

Generic functions that are never instantiated remain as `polang.generic_func` in MLIR. They are erased during lowering to standard dialects (since there's no concrete type to lower to).

```polang
let unused(x) = x  ; Kept as polang.generic_func, erased during lowering
42                 ; No call to unused, so no specialization created
```

## Error Handling

Error messages include source location information (line and column) to help identify where problems occur.

### Error Message Format

Type checker errors use the format:
```
Type error: <message> at line <line>, column <column>
```

MLIR-level errors use MLIR's location format:
```
loc("<source>":line:column): error: <message>
```

### Type Mismatch Errors

When types cannot be unified, the compiler reports an error:

```polang
let f(x: i64) = x
f(3.14)  ; Type error: argument 1 expects i64, got f64 at line 2, column 1
```

### Undeclared Variable Errors

References to undefined variables report the location of the reference:

```polang
x + 1  ; Type error: Undeclared variable: x at line 1, column 1
```

### Unresolved Type Variables

Generic functions without call sites remain as `polang.generic_func` with their type parameters unresolved. They are erased during lowering but can be instantiated in subsequent REPL inputs.
