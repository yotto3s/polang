# Polang Type System

This document describes the Polang type system, including type inference and polymorphic types.

## Table of Contents

- [Overview](#overview)
- [Primitive Types](#primitive-types)
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
- **No implicit conversions**: Types must match exactly (e.g., `int` and `double` are distinct)

The type system is implemented in two phases:

1. **Parser-level type checking**: Validates types and detects errors early; performs capture analysis for closures
2. **MLIR-level inference**: Resolves all type variables via unification (Hindley-Milner style)

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

### Boolean Type

| Type | Description | Size | MLIR Type | LLVM Type |
|------|-------------|------|-----------|-----------|
| `bool` | Boolean | 1-bit | `!polang.bool` | `i1` |

### Legacy Type Aliases

For compatibility, the following aliases are supported:

| Alias | Maps To |
|-------|---------|
| `int` | `i64` |
| `double` | `f64` |

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

  // Legacy aliases
  static constexpr const char* INT = "int";     // alias for i64
  static constexpr const char* DOUBLE = "double"; // alias for f64

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

## Type Inference

### Parser-Level Type Checking

The parser's type checker (`parser/src/type_checker.cpp`) focuses on **error detection** and **capture analysis**, while delegating type inference to MLIR. Untyped parameters are marked as `typevar` and resolved later via MLIR's unification algorithm.

**Type Checker Responsibilities:**

| Responsibility | Description |
|----------------|-------------|
| Error detection | Undefined variables, immutable reassignment, arity mismatch |
| Type validation | Validates explicit type annotations match usage |
| Capture analysis | Identifies free variables for closures via `FreeVariableCollector` |
| Type variable setup | Marks untyped parameters as `typevar` for MLIR inference |

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

All untyped parameters are marked as **type variables** by the parser. The MLIR type inference pass resolves these via unification, using constraints from:
- Literal values (e.g., `42` is `int`, `3.14` is `double`)
- Binary operations (operands must have same type)
- Call sites (argument types constrain parameter types)
- If conditions (must be `bool`)

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
MLIRGen: fn identity(%arg0: !polang.typevar<0>) -> !polang.typevar<0>
           ↓
Type Inference Pass: Unifies typevar<0> with int from call site
           ↓
Monomorphization: Creates identity$int specialized version
           ↓
Result: fn identity$int(%arg0: !polang.int) -> !polang.int
```

## Type Variables

Type variables represent unknown types that will be resolved during type inference.

### Representation

In the parser, type variables are represented as the string `"typevar"`:

```cpp
// When local inference fails, assign a type variable
mutable_arg.type = new NIdentifier(TypeNames::TYPEVAR);
```

In MLIR, type variables are parameterized types with unique IDs and a **kind** that constrains what types they can resolve to:

```mlir
!polang.typevar<0, Any>      ; Can be any type
!polang.typevar<1, Integer>  ; Must resolve to an integer type
!polang.typevar<2, Float>    ; Must resolve to a float type
```

### Type Variable Kinds

Type variables have a **kind** that constrains their resolution:

| Kind | Description | Default Resolution |
|------|-------------|-------------------|
| `Any` | No constraint, can be any type | `i64` (if no other info) |
| `Integer` | Must be an integer type (i8-i64, u8-u64) | `i64` |
| `Float` | Must be a float type (f32, f64) | `f64` |

The kind is inferred from usage:
- Integer literals (e.g., `42`) create `Integer`-kind constraints
- Float literals (e.g., `3.14`) create `Float`-kind constraints
- Operations on type variables propagate kind constraints

### Definition

Type variables are defined in `mlir/include/polang/Dialect/PolangTypes.td`:

```tablegen
def Polang_TypeVarKind : I32EnumAttr<"TypeVarKind", "Type variable kind", [
  I32EnumAttrCase<"Any", 0>,
  I32EnumAttrCase<"Integer", 1>,
  I32EnumAttrCase<"Float", 2>
]>;

def Polang_TypeVarType : Polang_Type<"TypeVar", "typevar"> {
  let summary = "Type variable for polymorphic types";
  let description = [{
    Represents an unresolved type variable used during type inference.
    The id is a unique identifier, and kind constrains resolution.
  }];
  let parameters = (ins "uint64_t":$id, "TypeVarKind":$kind);
  let assemblyFormat = "`<` $id `,` $kind `>`";
}
```

### Generation

The `MLIRGenVisitor` generates fresh type variables for untyped parameters:

```cpp
class MLIRGenVisitor : public Visitor {
  uint64_t nextTypeVarId = 0;

  Type freshTypeVar() {
    return builder.getType<TypeVarType>(nextTypeVarId++);
  }

  Type getTypeOrFresh(const NIdentifier* typeAnnotation) {
    if (typeAnnotation) {
      return getPolangType(typeAnnotation->name);
    }
    return freshTypeVar();
  }
};
```

## Unification Algorithm

The type inference pass uses the standard unification algorithm to resolve type variables.

### Substitution

A substitution maps type variable IDs to types:

```cpp
class Substitution {
  llvm::DenseMap<uint64_t, Type> bindings;
public:
  void bind(uint64_t var, Type type);
  [[nodiscard]] Type apply(Type type) const;      // Recursively resolve type vars
  [[nodiscard]] Substitution compose(const Substitution& other) const;
};
```

### Unifier

The unifier attempts to make two types equal:

```cpp
class Unifier {
public:
  bool unify(Type t1, Type t2, Substitution& subst) {
    // Apply current substitution first
    Type s1 = subst.apply(t1);
    Type s2 = subst.apply(t2);

    // Same type - trivially unifiable
    if (s1 == s2) return true;

    // Left is type variable - bind it
    if (auto var1 = dyn_cast<TypeVarType>(s1))
      return unifyVar(var1.getId(), s2, subst);

    // Right is type variable - bind it
    if (auto var2 = dyn_cast<TypeVarType>(s2))
      return unifyVar(var2.getId(), s1, subst);

    // Both concrete but different - cannot unify
    return false;
  }

private:
  bool unifyVar(uint64_t var, Type type, Substitution& subst) {
    // Occurs check - prevent infinite types
    if (occursIn(var, type)) return false;
    subst.bind(var, type);
    return true;
  }
};
```

### Constraint Collection

The type inference pass collects constraints from:

1. **Return statements**: Return value must match function return type
2. **Arithmetic operations**: Operands and result must have same type
3. **Call sites**: Argument types must match parameter types

```cpp
void collectFunctionConstraints(FuncOp func, Substitution& subst, Unifier& unifier) {
  Type expectedReturnType = func.getFunctionType().getResult(0);

  func.walk([&](ReturnOp returnOp) {
    Type actualType = returnOp.getValue().getType();
    unifier.unify(expectedReturnType, actualType, subst);
  });

  func.walk([&](Operation* op) {
    if (isa<AddOp, SubOp, MulOp, DivOp>(op)) {
      Type lhsType = op->getOperand(0).getType();
      Type rhsType = op->getOperand(1).getType();
      Type resultType = op->getResult(0).getType();
      unifier.unify(lhsType, rhsType, subst);
      unifier.unify(lhsType, resultType, subst);
    }
  });
}
```

## Implementation Details

### File Structure

```
parser/
├── include/parser/
│   ├── polang_types.hpp     # Type constants (INT, DOUBLE, BOOL, TYPEVAR)
│   └── type_checker.hpp     # Type checker interface
└── src/
    └── type_checker.cpp     # Error detection, capture analysis, typevar setup
                             # Contains: TypeChecker, FreeVariableCollector

mlir/
├── include/polang/
│   ├── Dialect/
│   │   ├── PolangTypes.td   # TypeVarType definition
│   │   └── PolangTypes.h    # Generated type classes
│   └── Transforms/
│       └── Passes.h         # Type inference & monomorphization pass declarations
└── lib/
    ├── Dialect/
    │   └── PolangTypes.cpp  # TypeVarType implementation
    └── Transforms/
        ├── TypeInference.cpp    # Unification and constraint solving
        └── Monomorphization.cpp # Function specialization for call sites
```

### Type Checking Flow

```
1. Parser builds AST
   └── Type annotations stored in NVariableDeclaration::type

2. Type checker runs (type_checker.cpp)
   ├── Validates explicit type annotations
   ├── Detects errors (undefined vars, mutability, arity)
   ├── Performs capture analysis (FreeVariableCollector)
   └── Marks untyped parameters as TYPEVAR

3. MLIRGen generates MLIR (MLIRGen.cpp)
   ├── Converts types to MLIR types (!polang.int, etc.)
   └── Generates !polang.typevar<id> for TYPEVAR

4. Type inference pass runs (TypeInference.cpp)
   ├── Collects constraints from operations and call sites
   ├── Unifies type variables with concrete types
   └── Updates operations with resolved types

5. Monomorphization pass runs (Monomorphization.cpp)
   ├── Creates specialized versions for each call site
   └── Updates calls to use specialized functions

6. Lowering proceeds with fully-typed MLIR
```

### Verifier Compatibility

Operation verifiers are updated to allow type variables during intermediate stages:

```cpp
// In PolangOps.cpp
namespace {
bool typesAreCompatible(Type t1, Type t2) {
  if (t1 == t2) return true;
  // Allow type variables - they will be resolved later
  if (isa<TypeVarType>(t1) || isa<TypeVarType>(t2)) return true;
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
- `y` is marked as `typevar` by the parser
- MLIR unifies `y` with `i64` (from `x + y` where `x: i64`)
- Return type inferred as `i64` (result of `+`)
- Monomorphization creates `add$i64_i64`

### Example 2: Polymorphic Identity

```polang
let identity(x) = x
identity(42)
```

**Inference:**
- `x` has no local constraints → assigned `typevar<0, Any>`
- Return type depends on `x` → assigned `typevar<1, Any>`
- Call `identity(42)` constrains `typevar<0>` to `i64` (integer literal default)
- Unification: `typevar<1>` = `typevar<0>` = `i64`
- Final type: `identity$i64: (i64) -> i64`

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
- `x` marked as `typevar` by parser
- MLIR unifies `x` with `Integer`-kind from `x > 0` where `0` is an integer literal
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
polang.func @identity(%arg0: !polang.typevar<0, Any>) -> !polang.typevar<1, Any>
    attributes {polang.polymorphic} { ... }

polang.func @identity$i64(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> { ... }

polang.func @identity$bool(%arg0: !polang.bool) -> !polang.bool { ... }
```

### Uncalled Polymorphic Functions

Polymorphic functions that are never called remain in MLIR with the `polang.polymorphic` attribute. They are skipped during lowering to LLVM IR (since there's no concrete type to lower to).

```polang
let unused(x) = x  ; Kept with polang.polymorphic attribute
42                 ; No call to unused, so no specialization created
```

## Error Handling

### Type Mismatch Errors

When types cannot be unified, the compiler reports an error:

```polang
let f(x: int) = x
f(3.14)  ; Error: argument 1 expects int, got double
```

### Unresolved Type Variables

Polymorphic functions without call sites keep their type variables and are marked with `polang.polymorphic`. They are skipped during lowering but can be called in subsequent REPL inputs.
