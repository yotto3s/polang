# MLIR Lowering Process

This document describes the MLIR-based code generation pipeline used by the Polang compiler.

## Overview

The Polang compiler uses MLIR (Multi-Level Intermediate Representation) for code generation. The compilation pipeline consists of multiple lowering stages:

```
Source Code
    ↓
   AST
    ↓
Polang Dialect (custom)
    ↓
Standard Dialects (arith, func, scf, memref)
    ↓
LLVM Dialect
    ↓
LLVM IR
```

## Polang Dialect

The Polang dialect is a custom MLIR dialect that closely mirrors the language semantics. This provides:

- **Extensibility**: Easy to add new language features
- **Optimization opportunities**: Custom passes can operate on high-level operations
- **Debugging**: The Polang dialect MLIR is human-readable and maps directly to source constructs

### Built-in Optimizations

**Immutable Variable SSA Optimization**: Immutable variables (`let`) are represented as SSA values directly, avoiding unnecessary memory allocation. This enables LLVM to perform constant folding and other optimizations more effectively.

### Types

| Polang Type | MLIR Type | Lowers To |
|-------------|-----------|-----------|
| `int` | `!polang.int` | `i64` |
| `double` | `!polang.double` | `f64` |
| `bool` | `!polang.bool` | `i1` |

### Operations

#### Constants

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.constant.int` | Integer literal | `%0 = polang.constant.int 42 : !polang.int` |
| `polang.constant.double` | Double literal | `%0 = polang.constant.double 3.14 : !polang.double` |
| `polang.constant.bool` | Boolean literal | `%0 = polang.constant.bool true : !polang.bool` |

#### Arithmetic

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.add` | Addition | `%2 = polang.add %0, %1 : !polang.int` |
| `polang.sub` | Subtraction | `%2 = polang.sub %0, %1 : !polang.int` |
| `polang.mul` | Multiplication | `%2 = polang.mul %0, %1 : !polang.int` |
| `polang.div` | Division | `%2 = polang.div %0, %1 : !polang.int` |

#### Comparison

| Operation | Predicates | Example |
|-----------|------------|---------|
| `polang.cmp` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` | `%2 = polang.cmp gt, %0, %1 : !polang.int` |

#### Functions

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.func` | Function definition | `polang.func @add(%a: !polang.int) -> !polang.int { ... }` |
| `polang.call` | Function call | `%0 = polang.call @add(%x) : (!polang.int) -> !polang.int` |
| `polang.return` | Return from function | `polang.return %0 : !polang.int` |

#### Control Flow

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.if` | If-then-else expression | `%0 = polang.if %cond -> !polang.int { ... } else { ... }` |
| `polang.yield` | Yield value from region | `polang.yield %0 : !polang.int` |

#### Variables

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.alloca` | Allocate mutable variable | `%0 = polang.alloca "x", mutable : !polang.int -> memref<i64>` |
| `polang.load` | Load from mutable variable | `%1 = polang.load %0 : memref<i64> -> !polang.int` |
| `polang.store` | Store to mutable variable | `polang.store %val, %0 : !polang.int, memref<i64>` |

**Note:** Immutable variables (declared with `let`) are optimized to use SSA values directly without memory allocation. Only mutable variables (declared with `let mut`) use the alloca/load/store pattern.

## Lowering Stages

### Stage 1: AST to Polang Dialect

The `MLIRGenVisitor` traverses the AST and generates Polang dialect operations. This stage:

- Creates `polang.func @__polang_entry` for the top-level code
- Generates nested functions for `let` bindings with function declarations
- Handles closure capture by adding captured variables as extra function parameters
- **Optimizes immutable variables** to use SSA values directly (no memory allocation)

#### Immutable Variable Optimization

Immutable variables (declared with `let`) are stored as SSA values and used directly without memory operations. This enables better optimization by LLVM (e.g., constant folding).

**Example (immutable variables):**

```polang
let x = 10
let y = 20
x + y
```

Generates:

```mlir
module {
  polang.func @__polang_entry() -> !polang.int {
    %0 = polang.constant.int 10 : !polang.int
    %1 = polang.constant.int 20 : !polang.int
    %2 = polang.add %0, %1 : !polang.int
    polang.return %2 : !polang.int
  }
}
```

#### Mutable Variable Handling

Mutable variables (declared with `let mut`) require memory allocation since their values can change:

**Example (mutable variable):**

```polang
let mut x = 10
x <- 20
x
```

Generates:

```mlir
module {
  polang.func @__polang_entry() -> !polang.int {
    %0 = polang.constant.int 10 : !polang.int
    %1 = polang.alloca "x", mutable : !polang.int -> memref<i64>
    polang.store %0, %1 : !polang.int, memref<i64>
    %2 = polang.constant.int 20 : !polang.int
    polang.store %2, %1 : !polang.int, memref<i64>
    %3 = polang.load %1 : memref<i64> -> !polang.int
    polang.return %3 : !polang.int
  }
}
```

### Stage 2: Polang to Standard Dialects

The `PolangToStandardPass` lowers Polang dialect operations to standard MLIR dialects:

| Polang Operation | Lowers To |
|------------------|-----------|
| `polang.constant.int` | `arith.constant` |
| `polang.constant.double` | `arith.constant` |
| `polang.constant.bool` | `arith.constant` |
| `polang.add` (int) | `arith.addi` |
| `polang.add` (double) | `arith.addf` |
| `polang.sub` (int) | `arith.subi` |
| `polang.sub` (double) | `arith.subf` |
| `polang.mul` (int) | `arith.muli` |
| `polang.mul` (double) | `arith.mulf` |
| `polang.div` (int) | `arith.divsi` |
| `polang.div` (double) | `arith.divf` |
| `polang.cmp` (int) | `arith.cmpi` |
| `polang.cmp` (double) | `arith.cmpf` |
| `polang.func` | `func.func` |
| `polang.call` | `func.call` |
| `polang.return` | `func.return` |
| `polang.if` | `scf.if` |
| `polang.yield` | `scf.yield` |
| `polang.alloca` | `memref.alloca` |
| `polang.load` | `memref.load` |
| `polang.store` | `memref.store` |

**Type Conversions:**

| Polang Type | Standard Type |
|-------------|---------------|
| `!polang.int` | `i64` |
| `!polang.double` | `f64` |
| `!polang.bool` | `i1` |

### Stage 3: Standard to LLVM Dialect

This stage uses built-in MLIR passes to lower standard dialects to the LLVM dialect:

1. `convert-scf-to-cf` - Converts `scf.if` to control flow with branches
2. `convert-func-to-llvm` - Converts `func.func` to `llvm.func`
3. `convert-arith-to-llvm` - Converts arithmetic operations
4. `convert-cf-to-llvm` - Converts control flow operations
5. `finalize-memref-to-llvm` - Converts memory operations
6. `reconcile-unrealized-casts` - Cleans up type conversion casts

### Stage 4: LLVM Dialect to LLVM IR

The final stage translates the LLVM dialect to actual LLVM IR using `translateModuleToLLVMIR`. This produces standard LLVM IR that can be:

- Printed as text (`.ll` files)
- Compiled to object code
- Executed via JIT

## Example: Complete Lowering

**Source:**
```polang
if 5 > 3 then 10 else 20
```

**Polang Dialect:**
```mlir
module {
  polang.func @__polang_entry() -> !polang.int {
    %0 = polang.constant.int 5 : !polang.int
    %1 = polang.constant.int 3 : !polang.int
    %2 = polang.cmp gt, %0, %1 : !polang.int
    %3 = polang.if %2 -> !polang.int {
      %4 = polang.constant.int 10 : !polang.int
      polang.yield %4 : !polang.int
    } else {
      %5 = polang.constant.int 20 : !polang.int
      polang.yield %5 : !polang.int
    }
    polang.return %3 : !polang.int
  }
}
```

**After Standard Lowering:**
```mlir
module {
  func.func @__polang_entry() -> i64 {
    %c5 = arith.constant 5 : i64
    %c3 = arith.constant 3 : i64
    %cmp = arith.cmpi sgt, %c5, %c3 : i64
    %result = scf.if %cmp -> i64 {
      %c10 = arith.constant 10 : i64
      scf.yield %c10 : i64
    } else {
      %c20 = arith.constant 20 : i64
      scf.yield %c20 : i64
    }
    return %result : i64
  }
}
```

**LLVM IR:**
```llvm
define i64 @__polang_entry() {
  %1 = icmp sgt i64 5, 3
  %2 = select i1 %1, i64 10, i64 20
  ret i64 %2
}
```

## Closure Handling

Closures are implemented by converting captured variables into extra function parameters. The `polang.func` operation has a `captures` attribute that lists captured variable names.

**Example:**
```polang
let base = 10 in
  let add_base = fn(x: int): int -> x + base in
    add_base(5)
```

The function `add_base` captures `base`. In MLIR:

```mlir
polang.func @add_base(%x: !polang.int, %base: !polang.int) -> !polang.int
    captures ["base"] {
  %0 = polang.add %x, %base : !polang.int
  polang.return %0 : !polang.int
}
```

At the call site, the captured value is passed as an extra argument:

```mlir
%result = polang.call @add_base(%arg, %base_val) : (!polang.int, !polang.int) -> !polang.int
```

## Compiler Flags

| Flag | Description |
|------|-------------|
| (default) | Output LLVM IR |
| `--emit-mlir` | Output Polang dialect MLIR |

## File Structure

```
mlir/
├── include/polang/
│   ├── Dialect/
│   │   ├── PolangDialect.td    # Dialect definition
│   │   ├── PolangOps.td        # Operation definitions
│   │   ├── PolangTypes.td      # Type definitions
│   │   └── *.h                 # Generated headers
│   ├── Conversion/
│   │   └── Passes.h            # Lowering pass declarations
│   └── MLIRGen.h               # AST to MLIR interface
└── lib/
    ├── Dialect/
    │   ├── PolangDialect.cpp   # Dialect implementation
    │   ├── PolangOps.cpp       # Operation implementations
    │   └── PolangTypes.cpp     # Type implementations
    ├── Conversion/
    │   └── PolangToStandard.cpp # Lowering pass
    └── MLIRGen/
        └── MLIRGen.cpp         # AST to MLIR visitor
```
