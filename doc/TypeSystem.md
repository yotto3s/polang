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

1. **Parser-level inference**: Infers types from local usage within function bodies
2. **MLIR-level inference**: Resolves polymorphic type variables from call sites

## Primitive Types

Polang supports three primitive types:

| Type | Description | Size | MLIR Type | LLVM Type |
|------|-------------|------|-----------|-----------|
| `int` | Signed integer | 64-bit | `!polang.int` | `i64` |
| `double` | Floating-point | 64-bit | `!polang.double` | `f64` |
| `bool` | Boolean | 1-bit | `!polang.bool` | `i1` |

### Type Constants

Type names are defined as compile-time constants in `parser/include/parser/polang_types.hpp`:

```cpp
struct TypeNames {
  static constexpr const char* INT = "int";
  static constexpr const char* DOUBLE = "double";
  static constexpr const char* BOOL = "bool";
  static constexpr const char* FUNCTION = "function";
  static constexpr const char* TYPEVAR = "typevar";
  static constexpr const char* UNKNOWN = "unknown";
};
```

## Type Inference

### Local Type Inference

The parser's type checker (`parser/src/type_checker.cpp`) performs local type inference within function bodies. It analyzes how parameters are used to determine their types.

**Inference Rules:**

| Usage Pattern | Inferred Type |
|---------------|---------------|
| `x + 1`, `x * 2` (with int literal) | `int` |
| `x + 1.0`, `x / 2.0` (with double literal) | `double` |
| `if x then ...` (as condition) | `bool` |
| `if cond then a else b` (result) | type of branches (must match) |
| `x + y` (where `y` has known type) | same as `y` |
| `f(x)` (where `f` expects a type) | parameter type of `f` |

**Example:**

```polang
let double_it(x) = x * 2    ; x inferred as int (from * 2)
let half(x) = x / 2.0       ; x inferred as double (from / 2.0)
let negate(x) = 0 - x       ; x inferred as int (from 0 - x)
```

**Implementation:**

The `ParameterTypeInferrer` class in `type_checker.cpp` traverses function bodies to collect type constraints:

```cpp
class ParameterTypeInferrer : public Visitor {
  // Tracks inferred types for each parameter
  std::map<std::string, std::string> inferred_types_;

  void visit(const NBinaryOperator& node) override {
    // If one operand is a parameter and the other has a known type,
    // infer the parameter's type from the known operand
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

### Polymorphic Type Inference

When local inference cannot determine a parameter's type (e.g., identity function `let id(x) = x`), the parameter is marked as a **type variable**. The MLIR type inference pass then resolves these from call sites.

**Example:**

```polang
let identity(x) = x     ; x cannot be locally inferred
identity(42)            ; call site constrains x to int
```

**Pipeline:**

```
Source: let identity(x) = x
           ↓
Parser: x marked as TYPEVAR
           ↓
MLIRGen: fn identity(%arg0: !polang.typevar<0>) -> !polang.typevar<1>
           ↓
Type Inference Pass: Unifies typevar<0> with int from call site
           ↓
Result: fn identity(%arg0: !polang.int) -> !polang.int
```

## Type Variables

Type variables represent unknown types that will be resolved during type inference.

### Representation

In the parser, type variables are represented as the string `"typevar"`:

```cpp
// When local inference fails, assign a type variable
mutable_arg.type = new NIdentifier(TypeNames::TYPEVAR);
```

In MLIR, type variables are parameterized types with unique IDs:

```mlir
!polang.typevar<0>   ; First type variable
!polang.typevar<1>   ; Second type variable
```

### Definition

Type variables are defined in `mlir/include/polang/Dialect/PolangTypes.td`:

```tablegen
def Polang_TypeVarType : Polang_Type<"TypeVar", "typevar"> {
  let summary = "Type variable for polymorphic types";
  let description = [{
    Represents an unresolved type variable used during type inference.
    The parameter is a unique identifier for the type variable.
  }];
  let parameters = (ins "uint64_t":$id);
  let assemblyFormat = "`<` $id `>`";
}
```

### Generation

The `MLIRGenVisitor` generates fresh type variables for untyped parameters:

```cpp
class MLIRGenVisitor : public Visitor {
  uint64_t nextTypeVarId_ = 0;

  Type freshTypeVar() {
    return builder.getType<TypeVarType>(nextTypeVarId_++);
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
  llvm::DenseMap<uint64_t, Type> bindings_;
public:
  void bind(uint64_t var, Type type);
  Type apply(Type type) const;      // Recursively resolve type vars
  Substitution compose(const Substitution& other) const;
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
    └── type_checker.cpp     # Local type inference implementation

mlir/
├── include/polang/
│   ├── Dialect/
│   │   ├── PolangTypes.td   # TypeVarType definition
│   │   └── PolangTypes.h    # Generated type classes
│   └── Transforms/
│       └── Passes.h         # Type inference pass declaration
└── lib/
    ├── Dialect/
    │   └── PolangTypes.cpp  # TypeVarType implementation
    └── Transforms/
        └── TypeInference.cpp # Unification and constraint solving
```

### Type Checking Flow

```
1. Parser builds AST
   └── Type annotations stored in NVariableDeclaration::type

2. Type checker runs (type_checker.cpp)
   ├── Validates explicit type annotations
   ├── Infers types from local usage
   └── Marks unresolved parameters as TYPEVAR

3. MLIRGen generates MLIR (MLIRGen.cpp)
   ├── Converts types to MLIR types (!polang.int, etc.)
   └── Generates !polang.typevar<id> for TYPEVAR

4. Type inference pass runs (TypeInference.cpp)
   ├── Collects constraints from operations
   ├── Unifies type variables with concrete types
   └── Updates operations with resolved types

5. Lowering proceeds with fully-typed MLIR
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
let add(x: int, y) = x + y
add(1, 2)
```

**Inference:**
- `x` is explicitly `int`
- `y` is used in `x + y` where `x: int`, so `y` inferred as `int`
- Return type inferred as `int` (result of `+`)

### Example 2: Polymorphic Identity

```polang
let identity(x) = x
identity(42)
```

**Inference:**
- `x` has no local constraints → assigned `typevar<0>`
- Return type depends on `x` → assigned `typevar<1>`
- Call `identity(42)` constrains `typevar<0>` to `int`
- Unification: `typevar<1>` = `typevar<0>` = `int`
- Final type: `identity: (int) -> int`

### Example 3: Multiple Parameters

```polang
let first(x, y) = x
first(1, 2.0)
```

**Inference:**
- `x` and `y` both get type variables
- Call site: `x` constrained to `int`, `y` constrained to `double`
- Final type: `first: (int, double) -> int`

### Example 4: Inference from Operations

```polang
let is_positive(x) = x > 0
is_positive(5)
```

**Inference:**
- `x > 0` where `0` is `int` → `x` inferred as `int`
- Return type is `bool` (result of comparison)
- Final type: `is_positive: (int) -> bool`

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
identity(42)     ; Creates identity$int
identity(true)   ; Creates identity$bool
```

**Generated MLIR:**

```mlir
polang.func @identity(%arg0: !polang.typevar<0>) -> !polang.typevar<1>
    attributes {polang.polymorphic} { ... }

polang.func @identity$int(%arg0: !polang.int) -> !polang.int { ... }

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
