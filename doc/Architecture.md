# Polang Architecture

This document describes the architecture and structure of the Polang compiler, including the MLIR-based code generation pipeline.

## Overview

Polang is a simple programming language compiler with an MLIR-based backend. It compiles source code to LLVM IR through a series of lowering passes.

## Project Structure

```
polang/
├── CMakeLists.txt              # Root CMake configuration
├── doc/                        # Documentation
│   ├── Syntax.md               # Language syntax reference
│   ├── TypeSystem.md           # Type system documentation
│   ├── PolangDialect.md        # MLIR dialect operations and types
│   ├── Architecture.md         # This file
│   ├── Building.md             # Build instructions
│   ├── Development.md          # Development guidelines
│   └── Testing.md              # Test and CI documentation
├── mlir/                       # MLIR dialect and code generation
│   ├── CMakeLists.txt
│   ├── include/polang/
│   │   ├── Dialect/            # Polang dialect definitions (.td files)
│   │   ├── Conversion/         # Lowering pass headers
│   │   ├── Transforms/         # Transform pass headers
│   │   └── MLIRGen.h           # AST to MLIR interface
│   └── lib/
│       ├── Dialect/            # Dialect implementation
│       ├── Conversion/         # PolangToStandard lowering pass
│       ├── Transforms/         # Type inference, monomorphization
│       └── MLIRGen/            # AST to MLIR visitor
├── parser/                     # Parser library
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── lexer.l             # Flex lexer
│   │   ├── parser.y            # Bison grammar
│   │   ├── parser_api.cpp      # Parser API implementation
│   │   ├── ast_printer.cpp     # AST printer implementation
│   │   ├── type_checker.cpp    # Type checking and inference
│   │   ├── operator_utils.cpp  # Operator utility functions
│   │   └── error_reporter.cpp  # Unified error reporting
│   └── include/parser/
│       ├── node.hpp            # AST node definitions
│       ├── parser_api.hpp      # Parser API header
│       ├── ast_printer.hpp     # AST printer header
│       ├── visitor.hpp         # Visitor pattern base class
│       ├── polang_types.hpp    # Type constants and utilities
│       ├── operator_utils.hpp  # Operator classification utilities
│       └── error_reporter.hpp  # Error reporting interface
├── compiler/                   # Compiler application
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp            # Compiler entry point
│   │   └── mlir_codegen.cpp    # MLIR-based code generation
│   └── include/compiler/
│       └── mlir_codegen.hpp    # MLIR code generation header
├── repl/                       # REPL application
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp            # REPL entry point
│   │   └── repl_session.cpp    # REPL session management
│   └── include/repl/
│       ├── repl_session.hpp    # REPL session header
│       └── input_checker.hpp   # Multi-line input detection
├── tests/                      # Test suite
│   ├── CMakeLists.txt          # Root test config (fetches GTest)
│   ├── common/
│   │   └── process_helper.hpp  # Shared test utilities
│   ├── parser/                 # Parser unit tests
│   ├── compiler/               # Compiler integration tests
│   ├── repl/                   # REPL tests
│   └── lit/                    # llvm-lit FileCheck tests
├── example/                    # Example programs
├── scripts/                    # Development scripts
│   ├── run-clang-format.sh     # Format source files
│   └── run-clang-tidy.sh       # Run static analysis
└── docker/                     # Docker build environment
```

## Components

### 1. Parser Library (`parser/`)

The parser library handles lexical analysis, parsing, and type checking.

- **Input**: Source code string via `polang_parse()` API
- **Output**: AST (`std::unique_ptr<NBlock>`)
- **Built as**: Static library `libPolangParser.a`

#### AST Ownership Model

The AST uses `std::unique_ptr` for memory management, ensuring automatic cleanup with no memory leaks:

- **Container ownership**: `StatementList`, `ExpressionList`, `VariableList`, and `LetBindingList` are all `std::vector<std::unique_ptr<T>>`, owning their elements
- **Node ownership**: Parent nodes own their children via `std::unique_ptr` (e.g., `NBinaryOperator` owns its `lhs` and `rhs` expressions)
- **Parser API**: `polang_parse()` returns `std::unique_ptr<NBlock>`, transferring ownership to the caller

**Example usage:**
```cpp
#include "parser/parser_api.hpp"

auto ast = polang_parse("let x = 5");
if (ast) {
    // Use ast->statements, etc.
    // Memory automatically freed when ast goes out of scope
}
```

#### Source Location Tracking

Every AST node carries source location information via the `SourceLocation` struct defined in `node.hpp`:

```cpp
struct SourceLocation {
  int line = 0;
  int column = 0;
  SourceLocation() = default;
  SourceLocation(int l, int c) : line(l), column(c) {}
  [[nodiscard]] bool isValid() const { return line > 0; }
};
```

Locations are populated during parsing using Bison's `@$` syntax, which tracks the position of grammar rules. This enables precise error messages with line and column information throughout the compilation pipeline.

**Key modules:**

| Module | Description |
|--------|-------------|
| `polang_types.hpp` | Centralized type constants (`TypeNames::INT`, etc.) |
| `operator_utils.hpp` | Operator classification and string conversion |
| `error_reporter.hpp` | Unified error reporting across all components |
| `visitor.hpp` | Visitor pattern base class for AST traversal |

### 2. MLIR Dialect (`mlir/`)

Custom MLIR dialect that closely mirrors Polang language semantics.

- **Types**:
  - Integer: `!polang.integer<width, signedness>` (e.g., `!polang.integer<64, signed>` for `i64`)
  - Float: `!polang.float<width>` (e.g., `!polang.float<64>` for `f64`)
  - Boolean: `!polang.bool`
  - Type variable: `!polang.typevar<id, kind>` (e.g., `!polang.typevar<0, Integer>`)
- **Operations**: Constants, arithmetic, comparisons, control flow, functions
- **Passes**: Type inference, monomorphization, lowering to standard dialects

### 3. Compiler (`compiler/`)

The compiler application that produces LLVM IR.

- **Input**: Source from stdin or file
- **Output**: LLVM IR to stdout
- **Process**: Parse → Type check → MLIR generation → Lowering → LLVM IR

### 4. REPL (`repl/`)

Interactive read-eval-print loop with JIT execution.

- **Features**:
  - Multi-line input for incomplete expressions
  - Variable and function persistence across evaluations
  - LLVM OrcJIT for code execution

## Compilation Pipeline

Source locations are tracked throughout the pipeline, enabling precise error messages with line and column information.

```
Source Code (.po)
       │
       ▼
┌─────────────────┐
│  Lexer (Flex)   │  parser/src/lexer.l
└────────┬────────┘
         │ Tokens (with positions)
         ▼
┌─────────────────┐
│  Parser (Bison) │  parser/src/parser.y
└────────┬────────┘
         │ AST (with SourceLocation on each node)
         ▼
┌─────────────────┐
│  Type Checker   │  parser/src/type_checker.cpp
└────────┬────────┘  (reports errors with line:column)
         │ Typed AST
         ▼
┌─────────────────┐
│   MLIRGen       │  mlir/lib/MLIRGen/MLIRGen.cpp
└────────┬────────┘  (uses FileLineColLoc for MLIR ops)
         │ Polang Dialect MLIR (with type variables)
         ▼
┌─────────────────┐
│ Type Inference  │  mlir/lib/Transforms/TypeInference.cpp
└────────┬────────┘
         │ Polang Dialect MLIR (types resolved)
         ▼
┌─────────────────┐
│ Monomorphization│  mlir/lib/Transforms/Monomorphization.cpp
└────────┬────────┘
         │ Polang Dialect MLIR (specialized functions)
         ▼
┌─────────────────┐
│PolangToStandard │  mlir/lib/Conversion/PolangToStandard.cpp
└────────┬────────┘
         │ Standard Dialects (arith, func, scf, memref)
         ▼
┌─────────────────┐
│ Standard to LLVM│  Built-in MLIR passes
└────────┬────────┘
         │ LLVM Dialect
         ▼
┌─────────────────┐
│  LLVM IR Output │  translateModuleToLLVMIR
└─────────────────┘
```

## Polang MLIR Dialect

The Polang dialect is a custom MLIR dialect that closely mirrors the language semantics. For detailed documentation of all operations, types, and passes, see [PolangDialect.md](PolangDialect.md).

This provides:

- **Extensibility**: Easy to add new language features
- **Optimization opportunities**: Custom passes can operate on high-level operations
- **Debugging**: The Polang dialect MLIR is human-readable and maps directly to source constructs

### Built-in Optimizations

**Immutable Variable SSA Optimization**: Immutable variables (`let`) are represented as SSA values directly, avoiding unnecessary memory allocation. This enables LLVM to perform constant folding and other optimizations more effectively.

### MLIR Verification

The Polang dialect uses custom verifiers to catch type errors during compilation while supporting type variables for polymorphism:

| Operation | Verification |
|-----------|--------------|
| `polang.add`, `polang.sub`, `polang.mul`, `polang.div` | Custom verifier ensures operands and result have compatible types (allows type variables) |
| `polang.cmp` | Custom verifier ensures operands have compatible types |
| `polang.if` | Verifies condition is `!polang.bool`, both branches yield same type |
| `polang.return` | Verifies return value matches function signature |
| `polang.call` | Verifies function exists, arity matches, and argument types match |
| `polang.ref.store` | Verifies reference is mutable |

Type variables are resolved by the type inference pass before lowering to standard dialects.

### Types

#### Integer Types

| Polang Type | MLIR Type | Lowers To |
|-------------|-----------|-----------|
| `i8` | `!polang.integer<8, signed>` | `i8` |
| `i16` | `!polang.integer<16, signed>` | `i16` |
| `i32` | `!polang.integer<32, signed>` | `i32` |
| `i64` | `!polang.integer<64, signed>` | `i64` |
| `u8` | `!polang.integer<8, unsigned>` | `i8` |
| `u16` | `!polang.integer<16, unsigned>` | `i16` |
| `u32` | `!polang.integer<32, unsigned>` | `i32` |
| `u64` | `!polang.integer<64, unsigned>` | `i64` |

#### Float Types

| Polang Type | MLIR Type | Lowers To |
|-------------|-----------|-----------|
| `f32` | `!polang.float<32>` | `f32` |
| `f64` | `!polang.float<64>` | `f64` |

#### Other Types

| Polang Type | MLIR Type | Lowers To |
|-------------|-----------|-----------|
| `bool` | `!polang.bool` | `i1` |
| (type variable) | `!polang.typevar<id, kind>` | (resolved by type inference) |

**Legacy aliases:** `int` maps to `i64`, `double` maps to `f64`.

#### Type Variables

Type variables (`!polang.typevar<id, kind>`) represent unknown types during the initial MLIR generation phase. They are used for:

- Function parameters without explicit type annotations
- Return types that depend on polymorphic parameters

Type variables have two components:
- **id**: A unique numeric identifier (e.g., `0`, `1`, `2`)
- **kind**: A constraint on what types it can resolve to (`Any`, `Integer`, or `Float`)

Type variables are resolved by the type inference pass before lowering to standard dialects. Unresolved type variables default to `i64` for `Integer` kind and `f64` for `Float` kind.

### Operations

#### Constants

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.constant.int` | Integer literal | `%0 = polang.constant.int 42 : !polang.integer<64, signed>` |
| `polang.constant.float` | Float literal | `%0 = polang.constant.float 3.14 : !polang.float<64>` |
| `polang.constant.bool` | Boolean literal | `%0 = polang.constant.bool true : !polang.bool` |

Note: Integer and float constants use the specific type at their point of use. Default literals use `i64` and `f64` respectively.

#### Arithmetic

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.add` | Addition | `%2 = polang.add %0, %1 : !polang.integer<64, signed>` |
| `polang.sub` | Subtraction | `%2 = polang.sub %0, %1 : !polang.integer<64, signed>` |
| `polang.mul` | Multiplication | `%2 = polang.mul %0, %1 : !polang.integer<64, signed>` |
| `polang.div` | Division | `%2 = polang.div %0, %1 : !polang.integer<64, signed>` |

Arithmetic operations work with any integer or float type of the same width and signedness.

#### Comparison

| Operation | Predicates | Example |
|-----------|------------|---------|
| `polang.cmp` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` | `%2 = polang.cmp gt, %0, %1 : !polang.integer<64, signed>` |

**Note:** Comparison operations only support numeric types (integer and float). Boolean values cannot be compared with `==` or `!=`; use conditional logic instead.

#### Functions

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.func` | Function definition | `polang.func @add(%a: !polang.integer<64, signed>) -> !polang.integer<64, signed> { ... }` |
| `polang.call` | Function call | `%0 = polang.call @add(%x) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>` |
| `polang.return` | Return from function | `polang.return %0 : !polang.integer<64, signed>` |

#### Control Flow

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.if` | If-then-else expression | `%0 = polang.if %cond -> !polang.integer<64, signed> { ... } else { ... }` |
| `polang.yield` | Yield value from region | `polang.yield %0 : !polang.integer<64, signed>` |

#### Reference Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `polang.ref.create` | Create mutable/immutable reference | `%0 = polang.ref.create %val {is_mutable = true} : !polang.integer<64, signed> -> !polang.ref<!polang.integer<64, signed>, mutable>` |
| `polang.ref.deref` | Read from reference | `%1 = polang.ref.deref %0 : !polang.ref<!polang.integer<64, signed>, mutable> -> !polang.integer<64, signed>` |
| `polang.ref.store` | Write to mutable reference | `%1 = polang.ref.store %val, %0 : !polang.integer<64, signed>, !polang.ref<!polang.integer<64, signed>, mutable> -> !polang.integer<64, signed>` |

**Note:** Immutable variables (declared with `let`) are optimized to use SSA values directly without memory allocation. Only mutable variables (declared with `let x = mut value`) use reference operations.

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

Mutable variables (declared with `let x = mut value`) use reference operations since their values can change:

**Example (mutable variable):**

```polang
let x = mut 10
x <- 20
*x
```

Generates:

```mlir
module {
  polang.func @__polang_entry() -> !polang.integer<64, signed> {
    %0 = polang.constant.integer 10 : !polang.integer<64, signed>
    %1 = polang.ref.create %0 {is_mutable = true} : !polang.integer<64, signed> -> !polang.ref<!polang.integer<64, signed>, mutable>
    %2 = polang.constant.integer 20 : !polang.integer<64, signed>
    %3 = polang.ref.store %2, %1 : !polang.integer<64, signed>, !polang.ref<!polang.integer<64, signed>, mutable> -> !polang.integer<64, signed>
    %4 = polang.ref.deref %1 : !polang.ref<!polang.integer<64, signed>, mutable> -> !polang.integer<64, signed>
    polang.return %4 : !polang.integer<64, signed>
  }
}
```

### Stage 2: Type Inference Pass

The `TypeInferencePass` resolves type variables using Hindley-Milner style unification. This pass:

1. **Collects constraints** from:
   - Function return statements (return value type must match function return type)
   - Arithmetic operations (operands and result must have the same type)
   - Call sites (argument types must match parameter types)

2. **Unifies types** using the standard unification algorithm:
   - Type variables can be bound to concrete types or other type variables
   - Occurs check prevents infinite types
   - Constraints are solved to produce a substitution mapping

3. **Applies substitution** to resolve all type variables:
   - Function signatures are updated with concrete types
   - Operations are rebuilt with resolved types
   - Block arguments are updated to match new types

**Example:**

Before type inference:
```mlir
polang.func @identity(%arg0: !polang.typevar<0>) -> !polang.typevar<1> {
  polang.return %arg0 : !polang.typevar<0>
}
polang.func @__polang_entry() -> !polang.int {
  %0 = polang.constant.int 42 : !polang.int
  %1 = polang.call @identity(%0) : (!polang.int) -> !polang.typevar<1>
  polang.return %1 : !polang.typevar<1>
}
```

After type inference:
```mlir
polang.func @identity(%arg0: !polang.int) -> !polang.int {
  polang.return %arg0 : !polang.int
}
polang.func @__polang_entry() -> !polang.int {
  %0 = polang.constant.int 42 : !polang.int
  %1 = polang.call @identity(%0) : (!polang.int) -> !polang.int
  polang.return %1 : !polang.int
}
```

### Stage 3: Polang to Standard Dialects

The `PolangToStandardPass` lowers Polang dialect operations to standard MLIR dialects:

| Polang Operation | Lowers To |
|------------------|-----------|
| `polang.constant.int` | `arith.constant` |
| `polang.constant.float` | `arith.constant` |
| `polang.constant.bool` | `arith.constant` |
| `polang.add` (integer) | `arith.addi` |
| `polang.add` (float) | `arith.addf` |
| `polang.sub` (integer) | `arith.subi` |
| `polang.sub` (float) | `arith.subf` |
| `polang.mul` (integer) | `arith.muli` |
| `polang.mul` (float) | `arith.mulf` |
| `polang.div` (signed integer) | `arith.divsi` |
| `polang.div` (unsigned integer) | `arith.divui` |
| `polang.div` (float) | `arith.divf` |
| `polang.cmp` (signed integer) | `arith.cmpi` (signed predicates) |
| `polang.cmp` (unsigned integer) | `arith.cmpi` (unsigned predicates) |
| `polang.cmp` (float) | `arith.cmpf` |
| `polang.func` | `func.func` |
| `polang.call` | `func.call` |
| `polang.return` | `func.return` |
| `polang.if` | `scf.if` |
| `polang.yield` | `scf.yield` |
| `polang.ref.create` (mutable) | `memref.alloca` + `memref.store` |
| `polang.ref.create` (immutable) | passthrough |
| `polang.ref.deref` | `memref.load` |
| `polang.ref.store` | `memref.store` |

**Type Conversions:**

| Polang Type | Standard Type |
|-------------|---------------|
| `!polang.integer<8, signed>` | `i8` |
| `!polang.integer<16, signed>` | `i16` |
| `!polang.integer<32, signed>` | `i32` |
| `!polang.integer<64, signed>` | `i64` |
| `!polang.integer<8, unsigned>` | `i8` |
| `!polang.integer<16, unsigned>` | `i16` |
| `!polang.integer<32, unsigned>` | `i32` |
| `!polang.integer<64, unsigned>` | `i64` |
| `!polang.float<32>` | `f32` |
| `!polang.float<64>` | `f64` |
| `!polang.bool` | `i1` |

### Stage 4: Standard to LLVM Dialect

This stage uses built-in MLIR passes to lower standard dialects to the LLVM dialect:

1. `convert-scf-to-cf` - Converts `scf.if` to control flow with branches
2. `convert-func-to-llvm` - Converts `func.func` to `llvm.func`
3. `convert-arith-to-llvm` - Converts arithmetic operations
4. `convert-cf-to-llvm` - Converts control flow operations
5. `finalize-memref-to-llvm` - Converts memory operations
6. `reconcile-unrealized-casts` - Cleans up type conversion casts

### Stage 5: LLVM Dialect to LLVM IR

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
  polang.func @__polang_entry() -> !polang.integer<64, signed> {
    %0 = polang.constant.int 5 : !polang.integer<64, signed>
    %1 = polang.constant.int 3 : !polang.integer<64, signed>
    %2 = polang.cmp gt, %0, %1 : !polang.integer<64, signed>
    %3 = polang.if %2 -> !polang.integer<64, signed> {
      %4 = polang.constant.int 10 : !polang.integer<64, signed>
      polang.yield %4 : !polang.integer<64, signed>
    } else {
      %5 = polang.constant.int 20 : !polang.integer<64, signed>
      polang.yield %5 : !polang.integer<64, signed>
    }
    polang.return %3 : !polang.integer<64, signed>
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
  let add_base = fn(x: i64): i64 -> x + base in
    add_base(5)
```

The function `add_base` captures `base`. In MLIR:

```mlir
polang.func @add_base(%x: !polang.integer<64, signed>, %base: !polang.integer<64, signed>) -> !polang.integer<64, signed>
    captures ["base"] {
  %0 = polang.add %x, %base : !polang.integer<64, signed>
  polang.return %0 : !polang.integer<64, signed>
}
```

At the call site, the captured value is passed as an extra argument:

```mlir
%result = polang.call @add_base(%arg, %base_val) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>
```

## Key Types

| Type | Location | Description |
|------|----------|-------------|
| `NBlock` | `node.hpp` | Root AST node containing statements |
| `std::unique_ptr<NBlock>` | `parser_api.hpp` | Owned pointer returned by `polang_parse()` |
| `SourceLocation` | `node.hpp` | Source position (line/column) for error reporting |
| `MLIRCodeGenContext` | `mlir_codegen.hpp` | MLIR code generation context |
| `Visitor` | `visitor.hpp` | Base class for AST visitors |
| `ErrorReporter` | `error_reporter.hpp` | Unified error reporting |
| `TypeNames` | `polang_types.hpp` | Compile-time type name constants |

## Usage

```bash
# Compile source to LLVM IR
echo "let x = 5" | ./build/bin/PolangCompiler

# Dump AST (for debugging)
echo "let x = 5" | ./build/bin/PolangCompiler --dump-ast

# Output Polang dialect MLIR (for debugging)
echo "let x = 5" | ./build/bin/PolangCompiler --emit-mlir

# Execute source file via REPL
./build/bin/PolangRepl example/hello.po

# Interactive REPL session
./build/bin/PolangRepl
```

### Compiler Flags

| Flag | Description |
|------|-------------|
| (default) | Output LLVM IR |
| `--dump-ast` | Dump AST and exit (no code generation) |
| `--emit-mlir` | Output Polang dialect MLIR instead of LLVM IR |
| `--help` | Show help message |

## MLIR File Structure

```
mlir/
├── include/polang/
│   ├── Dialect/
│   │   ├── PolangDialect.td    # Dialect definition
│   │   ├── PolangOps.td        # Operation definitions
│   │   ├── PolangTypes.td      # Type definitions (including TypeVarType)
│   │   ├── Passes.h            # Dialect pass declarations
│   │   └── *.h                 # Generated headers
│   ├── Conversion/
│   │   └── Passes.h            # Lowering pass declarations
│   ├── Transforms/
│   │   ├── Passes.h            # Transform pass declarations
│   │   └── Passes.td           # TableGen pass definitions
│   └── MLIRGen.h               # AST to MLIR interface
└── lib/
    ├── Dialect/
    │   ├── PolangDialect.cpp   # Dialect implementation
    │   ├── PolangOps.cpp       # Operation implementations (includes verifiers)
    │   └── PolangTypes.cpp     # Type implementations
    ├── Conversion/
    │   └── PolangToStandard.cpp # Lowering pass
    ├── Transforms/
    │   ├── TypeInference.cpp   # Type inference pass (unification algorithm)
    │   └── Monomorphization.cpp # Monomorphization pass (function specialization)
    └── MLIRGen/
        └── MLIRGen.cpp         # AST to MLIR visitor
```

## Related Documentation

- [Syntax.md](Syntax.md) - Language syntax reference
- [TypeSystem.md](TypeSystem.md) - Type system and inference
- [PolangDialect.md](PolangDialect.md) - MLIR dialect operations and types
- [Building.md](Building.md) - Build instructions
- [Testing.md](Testing.md) - Test infrastructure and CI/CD
