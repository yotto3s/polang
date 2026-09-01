# Polang MLIR Dialect

This document provides comprehensive documentation of the Polang MLIR dialect, including all operations, types, and transformation passes.

## Overview

The Polang dialect is a custom MLIR dialect that closely mirrors the Polang language semantics. It serves as an intermediate representation between the AST and standard MLIR dialects, enabling:

- **High-level optimizations** - Operations at language level before lowering
- **Type inference** - Hindley-Milner style type resolution using type variables
- **Monomorphization** - Specialization of polymorphic functions
- **Debugging** - Human-readable IR that maps directly to source constructs

The dialect is defined using TableGen in `mlir/include/polang/Dialect/`.

## Types

The Polang compiler leverages standard built-in MLIR types for primitives and tuples (`mlir::TupleType`), and defines a dialect-specific type for type parameters (`TypeParamType`).

### Standard Primitive Types

| Polang Type | MLIR Built-in Type | LLVM Type | Description |
|-------------|-------------------|-----------|-------------|
| `i8` | `si8` | `i8` | Signed 8-bit integer |
| `i16` | `si16` | `i16` | Signed 16-bit integer |
| `i32` | `si32` | `i32` | Signed 32-bit integer |
| `i64` | `si64` | `i64` | Signed 64-bit integer |
| `u8` | `ui8` | `i8` | Unsigned 8-bit integer |
| `u16` | `ui16` | `i16` | Unsigned 16-bit integer |
| `u32` | `ui32` | `i32` | Unsigned 32-bit integer |
| `u64` | `ui64` | `i64` | Unsigned 64-bit integer |
| `isize` / `usize` | `index` | `i64` | Target index type |
| `f32` | `f32` | `f32` | Single-precision float |
| `f64` | `f64` | `f64` | Double-precision float |
| `bool` | `i1` | `i1` | Boolean true/false |

### Type Parameter Type

Type parameters represent named type variables in generic function templates. They use ML-style naming convention ('a, 'b, etc.) and are resolved during monomorphization when concrete types are known from call sites.

| MLIR Type | Description |
|-----------|-------------|
| `!polang.type_param<"a">` | Named type parameter "a" |
| `!polang.type_param<"b">` | Named type parameter "b" |

Type parameters appear in `polang.generic_func` signatures and are bound to concrete types by `polang.instantiate` operations at call sites.

### Tuple Types

Represents a fixed-size heterogeneous product of types with arbitrary arity ($N \ge 0$). Polang directly uses MLIR's built-in `mlir::TupleType` (`tuple<...>`). Elements can be standard MLIR types, nested `tuple`s, or type parameters (`TypeParamType`).

`DataLayoutTypeInterface` is registered for `mlir::TupleType` in the Polang dialect. Each element in a tuple occupies a 64-bit (8-byte) aligned slot in declaration order without re-ordering.

| Polang Type | MLIR Type | Description |
|-------------|-----------|-------------|
| `()` | `tuple<>` | 0-tuple (unit) |
| `(i64, f64)` | `tuple<si64, f64>` | 2-tuple of si64 and f64 |
| `(i64, (f64, bool))` | `tuple<si64, tuple<f64, i1>>` | Nested tuple |
| `('a, 'b)` | `tuple<!polang.type_param<"a">, !polang.type_param<"b">>` | Generic tuple |

### Type Constraints

- **`Polang_AnyNumericOrParam`**: `AnyInteger`, `AnyFloat`, `Index`, `TypeParamType`.
- **`Polang_AnyTypeOrParam`**: `AnyInteger`, `AnyFloat`, `Index`, `AnyTuple`, `TypeParamType`.
- **`Polang_I1OrParamType`**: `I1`, `TypeParamType`.

## Operations

### Constant Operations

#### `polang.constant.integer`

Produces a constant integer or index value.

```mlir
%0 = polang.constant.integer 42 : si64
%1 = polang.constant.integer 255 : ui8
```

#### `polang.constant.float`

Produces a constant float value.

```mlir
%0 = polang.constant.float 3.14 : f64
%1 = polang.constant.float 1.5 : f32
```

### Arithmetic Operations

All arithmetic operations require operands and result to have compatible types.

#### `polang.add`

Addition of two numeric values.

```mlir
%2 = polang.add %0, %1 : si64, si64 -> si64
```

#### `polang.sub`

Subtraction of second operand from first.

```mlir
%2 = polang.sub %0, %1 : si64, si64 -> si64
```

#### `polang.mul`

Multiplication of two numeric values.

```mlir
%2 = polang.mul %0, %1 : si64, si64 -> si64
```

#### `polang.div`

Division of first operand by second.

```mlir
%2 = polang.div %0, %1 : si64, si64 -> si64
```

### Comparison Operation

#### `polang.cmp`

Compares two numeric values and returns an `i1` result.

**Predicates:** `eq` (equal), `ne` (not equal), `lt` (less than), `le` (less or equal), `gt` (greater than), `ge` (greater or equal)

```mlir
%2 = polang.cmp eq, %0, %1 : si64, si64
%3 = polang.cmp gt, %a, %b : f64, f64
```

### Type Conversion

#### `polang.cast`

Explicit type conversion between numeric types.

**Supported conversions:**
- Integer to integer (widening/narrowing)
- Float to float (widening/narrowing)
- Integer to float
- Float to integer (saturating truncation toward zero)
- Index to/from integer and float

```mlir
%0 = polang.cast %x : si32 -> si64
%1 = polang.cast %f : f64 -> si32
```

### Function Operations

#### `polang.func`

Defines a function with a name, type signature, and body.

```mlir
polang.func @add(%a: si64, %b: si64) -> si64 {
  %c = polang.add %a, %b : si64, si64 -> si64
  polang.return %c : si64
}
```

**Attributes:**
- `sym_name` - Function name (symbol)
- `function_type` - Function type signature
- `captures` - Optional list of captured variable names (for closures)

#### `polang.call`

Calls a function with the given arguments.

```mlir
%0 = polang.call @add(%a, %b) : (si64, si64) -> si64
```

#### `polang.return`

Returns a value from a function.

```mlir
polang.return %0 : si64
```

### Control Flow Operations

Control flow expressions (such as `if-then-else` and short-circuit `&&` / `||`) directly use standard MLIR operations (`scf.if` and `scf.yield`).

```mlir
%0 = scf.if %cond -> (si64) {
  scf.yield %a : si64
} else {
  scf.yield %b : si64
}
```

#### `polang.yield`

Yields a value from a `polang.let_expr` body region.

```mlir
polang.yield %0 : si64
```

### Generic Function Operations

#### `polang.generic_func`

Defines a polymorphic function with named type parameters and optional trait bounds. Type parameters use ML-style naming ('a, 'b, etc.).

```mlir
polang.generic_func @identity<a>(%arg0: !polang.type_param<"a">) -> !polang.type_param<"a"> {
  polang.return %arg0 : !polang.type_param<"a">
}
```

**Attributes:**
- `sym_name` - Function name (symbol)
- `function_type` - Function type signature
- `type_params` - List of type parameter names (e.g., `["a", "b"]`)
- `type_param_bounds` - Optional bounds for type parameters
- `captures` - Optional list of captured variable names (for closures)

#### `polang.instantiate`

Calls a generic function with concrete type bindings for its type parameters.

```mlir
%0 = polang.instantiate @identity<a = si64>(%x) : (si64) -> si64
```

**Attributes:**
- `callee` - Symbol reference to the generic function
- `type_param_names` - Names of type parameters being bound
- `type_bindings` - Concrete types to bind to each parameter

### Let Expression Operations

#### `polang.let_expr`

A let expression with separate regions for each binding and a body. Each binding region computes a value independently and yields it via `polang.yield.binding`. The body region receives bound values as block arguments and yields via `polang.yield`.

```mlir
%0 = polang.let_expr ["x", "y"] -> si64
  binding {
    polang.yield.binding %a : si64
  }
  binding {
    polang.yield.binding %b : si64
  }
  body {
  ^bb0(%x: si64, %y: si64):
    %sum = polang.add %x, %y : si64, si64 -> si64
    polang.yield %sum : si64
  }
```

**Attributes:**
- `var_names` - Array of binding variable names

#### `polang.yield.binding`

Terminates a binding region of a let expression and yields the bound value as input to the body region's corresponding block argument.

```mlir
polang.yield.binding %0 : si64
```

### Global Variable Operations

#### `polang.global`

Declares a module-level global variable. Used in the REPL for cross-module variable persistence. When `is_external` is set, the global is an extern declaration (defined in a previously compiled module; JIT resolves the symbol). Non-external globals may have an optional initializer region that computes the initial value.

```mlir
polang.global @x : si64 {
  %v = polang.constant.integer 42 : si64
  polang.yield.global %v : si64
}
polang.global @y {is_external} : si64
```

**Attributes:**
- `sym_name` - Global variable name
- `type` - Type of the global variable
- `is_external` - Unit attribute marking extern declarations

**Regions:**
- `initializer` - Optional region (terminated by `polang.yield.global`) that computes the initial value

#### `polang.yield.global`

Yields the initial value from a `polang.global` initializer region.

```mlir
polang.yield.global %v : si64
```

#### `polang.global.load`

Loads the current value from a previously declared global variable.

```mlir
%0 = polang.global.load @x : si64
```

### Debug Operations

#### `polang.print`

Prints a value to stdout (for debugging and output).

```mlir
polang.print %0 : si64
```

## Operation Verifiers

The Polang dialect uses custom verifiers to catch type errors during compilation while supporting type variables for polymorphism.

| Operation | Verification |
|-----------|--------------|
| `polang.add`, `polang.sub`, `polang.mul`, `polang.div` | Operands and result must have compatible types (allows type parameters) |
| `polang.cmp` | Operands must have compatible types |
| `polang.cast` | Source and target must be numeric or index types |
| `polang.return` | Return value type must match function signature |
| `polang.call` | Function must exist, arity and argument types must match |

## Transformation Passes

### Monomorphization Pass (`polang-monomorphize`)

Specializes polymorphic functions for each unique set of concrete type arguments.

**Process:**
1. Identify generic functions (`polang.generic_func` with type parameters)
2. Analyze call sites to collect concrete type combinations
3. Create specialized function copies for each combination
4. Update calls to use specialized versions
5. Apply name mangling: `func$type1$type2`

**Example:**

Source:
```polang
identity(x) = x
identity(42)
identity(true)
```

After monomorphization:
```mlir
polang.func @identity$i64(%arg0: si64) -> si64 { ... }
polang.func @identity$bool(%arg0: i1) -> i1 { ... }
```

### PolangToStandard Pass (`polang-to-standard`)

Lowers Polang dialect operations to standard MLIR dialects (arith, func, scf, memref, LLVM).

**Operation Lowering:**

| Polang Operation | Lowers To |
|------------------|-----------|
| `polang.constant.integer` | `arith.constant` |
| `polang.constant.float` | `arith.constant` |
| `polang.add` (integer) | `arith.addi` |
| `polang.add` (float) | `arith.addf` |
| `polang.sub` (integer) | `arith.subi` |
| `polang.sub` (float) | `arith.subf` |
| `polang.mul` (integer) | `arith.muli` |
| `polang.mul` (float) | `arith.mulf` |
| `polang.div` (signed int) | `arith.divsi` |
| `polang.div` (unsigned int) | `arith.divui` |
| `polang.div` (float) | `arith.divf` |
| `polang.cmp` (signed int) | `arith.cmpi` (slt, sle, sgt, sge) |
| `polang.cmp` (unsigned int) | `arith.cmpi` (ult, ule, ugt, uge) |
| `polang.cmp` (float) | `arith.cmpf` |
| `polang.func` | `func.func` |
| `polang.call` | `func.call` |
| `polang.return` | `func.return` |
| `polang.generic_func` | Erased (must be monomorphized first) |
| `polang.instantiate` | Erased (replaced by `polang.call` during monomorphization) |
| `polang.global` | `memref.global` (0-d memref) |
| `polang.global.load` | `memref.get_global` + `memref.load` |

**Type Lowering:**

| Polang MLIR Type | Standard Lowered Type |
|------------------|----------------------|
| `siN` / `uiN` | `iN` (signless) |
| `f32` | `f32` |
| `f64` | `f64` |
| `i1` | `i1` |
| `index` | `index` |

## File Structure

```
mlir/
├── include/polang/
│   ├── Dialect/
│   │   ├── PolangDialect.td    # Dialect definition
│   │   ├── PolangOps.td        # Operation definitions
│   │   ├── PolangTypes.td      # Type definitions
│   │   ├── PolangEnums.td      # Enum definitions
│   │   └── PolangLocations.h   # Custom location types
│   ├── Conversion/
│   │   └── Passes.h            # Lowering pass declarations
│   ├── Transforms/
│   │   ├── Passes.h            # Transform pass declarations
│   │   └── Passes.td           # TableGen pass definitions
│   ├── MLIRGen.h               # AST to MLIR interface
│   └── PolangTypeConverter.h   # Polang-to-standard type converter
├── lib/
│   ├── Dialect/
│   │   ├── PolangDialect.cpp       # Dialect implementation
│   │   ├── PolangOps.cpp           # Operation implementations (verifiers)
│   │   └── PolangTypes.cpp         # Type implementations
│   ├── Conversion/
│   │   └── PolangToStandard.cpp    # Lowering pass
│   ├── Transforms/
│   │   └── Monomorphization.cpp    # Monomorphization pass
│   └── MLIRGen/
│       ├── MLIRGen.cpp             # AST to MLIR visitor
│       ├── PolangLocations.cpp     # Custom location implementations
│       └── PolangTypeConverter.cpp # Type converter implementation
└── tools/
    └── polang-opt/
        └── polang-opt.cpp          # MLIR round-trip testing tool
```

## Related Documentation

- [Architecture.md](Architecture.md) - Compilation pipeline overview
- [TypeSystem.md](TypeSystem.md) - Type system and inference details
- [Syntax.md](Syntax.md) - Language syntax reference
