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

### Integer Types

Parameterized integer type with configurable width and signedness.

| Polang Type | MLIR Type | LLVM Type | Description |
|-------------|-----------|-----------|-------------|
| `i8` | `!polang.integer<8, signed>` | `i8` | Signed 8-bit integer |
| `i16` | `!polang.integer<16, signed>` | `i16` | Signed 16-bit integer |
| `i32` | `!polang.integer<32, signed>` | `i32` | Signed 32-bit integer |
| `i64` | `!polang.integer<64, signed>` | `i64` | Signed 64-bit integer |
| `u8` | `!polang.integer<8, unsigned>` | `i8` | Unsigned 8-bit integer |
| `u16` | `!polang.integer<16, unsigned>` | `i16` | Unsigned 16-bit integer |
| `u32` | `!polang.integer<32, unsigned>` | `i32` | Unsigned 32-bit integer |
| `u64` | `!polang.integer<64, unsigned>` | `i64` | Unsigned 64-bit integer |

### Float Types

Parameterized floating-point type with configurable width.

| Polang Type | MLIR Type | LLVM Type | Description |
|-------------|-----------|-----------|-------------|
| `f32` | `!polang.float<32>` | `f32` | Single-precision float |
| `f64` | `!polang.float<64>` | `f64` | Double-precision float |

### Boolean Type

| Polang Type | MLIR Type | LLVM Type | Description |
|-------------|-----------|-----------|-------------|
| `bool` | `!polang.bool` | `i1` | Boolean true/false |

### Type Variable Type

Type variables represent unknown types that are resolved during type inference.

| MLIR Type | Description |
|-----------|-------------|
| `!polang.typevar<id>` | Unconstrained type variable |
| `!polang.typevar<id, any>` | Can unify with any type (default) |
| `!polang.typevar<id, integer>` | Must resolve to an integer type |
| `!polang.typevar<id, float>` | Must resolve to a float type |

**Type Variable Kinds:**

| Kind | Description | Default Resolution |
|------|-------------|-------------------|
| `any` | No constraint | `i64` |
| `integer` | Must be integer type (i8-i64, u8-u64) | `i64` |
| `float` | Must be float type (f32, f64) | `f64` |

### Reference Types

Reference types wrap a value and provide read/write access.

| Polang Type | MLIR Type | Description |
|-------------|-----------|-------------|
| `mut T` | `!polang.ref<T, mutable>` | Mutable reference (read/write) |
| `ref T` | `!polang.ref<T>` | Immutable reference (read-only) |

**Examples:**
```mlir
!polang.ref<!polang.integer<64, signed>, mutable>  ; mutable ref i64
!polang.ref<!polang.integer<64, signed>>           ; immutable ref i64
```

## Operations

### Constant Operations

#### `polang.constant.integer`

Produces a constant integer value.

```mlir
%0 = polang.constant.integer 42 : !polang.integer<64, signed>
%1 = polang.constant.integer 255 : !polang.integer<8, unsigned>
```

#### `polang.constant.float`

Produces a constant float value.

```mlir
%0 = polang.constant.float 3.14 : !polang.float<64>
%1 = polang.constant.float 1.5 : !polang.float<32>
```

#### `polang.constant.bool`

Produces a constant boolean value.

```mlir
%0 = polang.constant.bool true : !polang.bool
%1 = polang.constant.bool false : !polang.bool
```

### Arithmetic Operations

All arithmetic operations require operands and result to have compatible types.

#### `polang.add`

Addition of two numeric values.

```mlir
%2 = polang.add %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
```

#### `polang.sub`

Subtraction of second operand from first.

```mlir
%2 = polang.sub %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
```

#### `polang.mul`

Multiplication of two numeric values.

```mlir
%2 = polang.mul %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
```

#### `polang.div`

Division of first operand by second.

```mlir
%2 = polang.div %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
```

### Comparison Operation

#### `polang.cmp`

Compares two numeric values and returns a boolean result.

**Predicates:** `eq` (equal), `ne` (not equal), `lt` (less than), `le` (less or equal), `gt` (greater than), `ge` (greater or equal)

```mlir
%2 = polang.cmp eq, %0, %1 : !polang.integer<64, signed>, !polang.integer<64, signed>
%3 = polang.cmp gt, %a, %b : !polang.float<64>, !polang.float<64>
```

### Type Conversion

#### `polang.cast`

Explicit type conversion between numeric types.

**Supported conversions:**
- Integer to integer (widening/narrowing)
- Float to float (widening/narrowing)
- Integer to float
- Float to integer (saturating truncation toward zero)

```mlir
%0 = polang.cast %x : !polang.integer<32, signed> -> !polang.integer<64, signed>
%1 = polang.cast %f : !polang.float<64> -> !polang.integer<32, signed>
```

### Function Operations

#### `polang.func`

Defines a function with a name, type signature, and body.

```mlir
polang.func @add(%a: !polang.integer<64, signed>, %b: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  %c = polang.add %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
  polang.return %c : !polang.integer<64, signed>
}
```

**Attributes:**
- `sym_name` - Function name (symbol)
- `function_type` - Function type signature
- `captures` - Optional list of captured variable names (for closures)

#### `polang.call`

Calls a function with the given arguments.

```mlir
%0 = polang.call @add(%a, %b) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>
```

#### `polang.return`

Returns a value from a function.

```mlir
polang.return %0 : !polang.integer<64, signed>
```

### Control Flow Operations

#### `polang.if`

If-then-else expression. Both branches must yield a value of the same type.

```mlir
%0 = polang.if %cond : !polang.bool -> !polang.integer<64, signed> {
  polang.yield %a : !polang.integer<64, signed>
} else {
  polang.yield %b : !polang.integer<64, signed>
}
```

#### `polang.yield`

Yields a value from a region (used within if-then-else branches).

```mlir
polang.yield %0 : !polang.integer<64, signed>
```

### Variable Operations

#### `polang.alloca`

Allocates stack memory for a mutable variable.

```mlir
%0 = polang.alloca "x", mutable : !polang.integer<64, signed> -> memref<!polang.integer<64, signed>>
```

**Note:** Immutable variables (`let`) are optimized to use SSA values directly without allocation.

#### `polang.load`

Loads a value from a memory location.

```mlir
%0 = polang.load %ref : memref<i64> -> !polang.integer<64, signed>
```

#### `polang.store`

Stores a value to a memory location. The verifier ensures the target is mutable.

```mlir
polang.store %value, %ref : !polang.integer<64, signed>, memref<i64>
```

### Reference Operations

#### `polang.ref.create`

Creates a reference from an initial value or source reference.

```mlir
; Create mutable reference (allocates memory)
%ref = polang.ref.create %val {is_mutable = true} : !polang.integer<64, signed> -> !polang.ref<!polang.integer<64, signed>, mutable>

; Create immutable reference from mutable (no allocation)
%imm = polang.ref.create %mut {is_mutable = false} : !polang.ref<!polang.integer<64, signed>, mutable> -> !polang.ref<!polang.integer<64, signed>>
```

#### `polang.ref.deref`

Dereferences a reference to read the value.

```mlir
%val = polang.ref.deref %ref : !polang.ref<!polang.integer<64, signed>, mutable> -> !polang.integer<64, signed>
```

#### `polang.ref.store`

Stores a value through a mutable reference. The verifier ensures the reference is mutable.

```mlir
%result = polang.ref.store %val, %ref : !polang.integer<64, signed>, !polang.ref<!polang.integer<64, signed>, mutable> -> !polang.integer<64, signed>
```

### Debug Operations

#### `polang.print`

Prints a value to stdout (for debugging and output).

```mlir
polang.print %0 : !polang.integer<64, signed>
```

## Operation Verifiers

The Polang dialect uses custom verifiers to catch type errors during compilation while supporting type variables for polymorphism.

| Operation | Verification |
|-----------|--------------|
| `polang.add`, `polang.sub`, `polang.mul`, `polang.div` | Operands and result must have compatible types (allows type variables) |
| `polang.cmp` | Operands must have compatible types |
| `polang.cast` | Source and target must be numeric types |
| `polang.if` | Condition must be bool, branches must yield same type |
| `polang.return` | Return value type must match function signature |
| `polang.call` | Function must exist, arity and argument types must match |
| `polang.store` | Target must be from a mutable allocation |
| `polang.ref.store` | Reference must be mutable (`!polang.ref<T, mutable>`) |

**Type Compatibility:** Operations use a `typesAreCompatible()` helper that allows type variables during intermediate stages (before type inference resolves them).

## Transformation Passes

### Type Inference Pass (`polang-type-inference`)

Performs Hindley-Milner style type inference to resolve type variables.

**Pipeline:**
1. **Collect constraints** from operations:
   - Return statements: return value must match function return type
   - Arithmetic operations: operands and result must have same type
   - Call sites: argument types must match parameter types

2. **Solve constraints** using unification algorithm:
   - Type variables can be bound to concrete types or other type variables
   - Occurs check prevents infinite types

3. **Apply substitution** to resolve all type variables:
   - Function signatures updated with concrete types
   - Operations rebuilt with resolved types

**Example transformation:**

Before:
```mlir
polang.func @identity(%arg0: !polang.typevar<0>) -> !polang.typevar<1> {
  polang.return %arg0 : !polang.typevar<0>
}
%0 = polang.call @identity(%x) : (!polang.integer<64, signed>) -> !polang.typevar<1>
```

After:
```mlir
polang.func @identity(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
  polang.return %arg0 : !polang.integer<64, signed>
}
%0 = polang.call @identity(%x) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
```

### Monomorphization Pass (`polang-monomorphize`)

Specializes polymorphic functions for each unique set of concrete type arguments.

**Process:**
1. Identify polymorphic functions (those with type variables or marked `polang.polymorphic`)
2. Analyze call sites to collect concrete type combinations
3. Create specialized function copies for each combination
4. Update calls to use specialized versions
5. Apply name mangling: `func$type1$type2`

**Example:**

Source:
```polang
let identity(x) = x
identity(42)
identity(true)
```

After monomorphization:
```mlir
polang.func @identity$i64(%arg0: !polang.integer<64, signed>) -> !polang.integer<64, signed> { ... }
polang.func @identity$bool(%arg0: !polang.bool) -> !polang.bool { ... }
```

### PolangToStandard Pass (`polang-to-standard`)

Lowers Polang dialect operations to standard MLIR dialects (arith, func, scf, memref).

**Operation Lowering:**

| Polang Operation | Lowers To |
|------------------|-----------|
| `polang.constant.integer` | `arith.constant` |
| `polang.constant.float` | `arith.constant` |
| `polang.constant.bool` | `arith.constant` |
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
| `polang.if` | `scf.if` |
| `polang.yield` | `scf.yield` |
| `polang.alloca` | `memref.alloca` |
| `polang.load` | `memref.load` |
| `polang.store` | `memref.store` |

**Type Lowering:**

| Polang Type | Standard Type |
|-------------|---------------|
| `!polang.integer<N, signed/unsigned>` | `iN` |
| `!polang.float<32>` | `f32` |
| `!polang.float<64>` | `f64` |
| `!polang.bool` | `i1` |
| `!polang.ref<T, mutable>` | `memref<T>` |

## File Structure

```
mlir/
├── include/polang/
│   ├── Dialect/
│   │   ├── PolangDialect.td    # Dialect definition
│   │   ├── PolangOps.td        # Operation definitions
│   │   ├── PolangTypes.td      # Type definitions
│   │   └── Passes.h            # Dialect pass declarations
│   ├── Conversion/
│   │   └── Passes.h            # Lowering pass declarations
│   ├── Transforms/
│   │   ├── Passes.h            # Transform pass declarations
│   │   └── Passes.td           # TableGen pass definitions
│   └── MLIRGen.h               # AST to MLIR interface
└── lib/
    ├── Dialect/
    │   ├── PolangDialect.cpp   # Dialect implementation
    │   ├── PolangOps.cpp       # Operation implementations (verifiers)
    │   └── PolangTypes.cpp     # Type implementations
    ├── Conversion/
    │   └── PolangToStandard.cpp # Lowering pass
    ├── Transforms/
    │   ├── TypeInference.cpp   # Type inference pass
    │   └── Monomorphization.cpp # Monomorphization pass
    └── MLIRGen/
        └── MLIRGen.cpp         # AST to MLIR visitor
```

## Related Documentation

- [Architecture.md](Architecture.md) - Compilation pipeline overview
- [TypeSystem.md](TypeSystem.md) - Type system and inference details
- [Syntax.md](Syntax.md) - Language syntax reference
