# Arrays, Slices, and Mutability Design

## Summary

Add arrays, slices, loop comprehensions, mutability, and a full polyhedral optimization pipeline to polang using MLIR's Affine dialect. This extends issue #33 with dynamically-sized slices and a coherent mutability model.

## Types and Memory Model

### Fixed Array `[T; N]`

- Size known at compile time
- Stack-allocated via `memref.alloca`
- MLIR type: `memref<N x T>`

### Slice `[T]`

- Size fixed at creation, known at runtime (ptr + length)
- Heap-allocated via `memref.alloc`
- MLIR type: `memref<? x T>`
- Not growable — a future `Vec` type will handle growable containers
- Future: ownership-based buffer deallocation will handle freeing

### Type Inference Rules

| Expression | Inferred Type |
|-----------|---------------|
| `[1, 2, 3]` | `[i64; 3]` |
| `let a: [i64] = [1, 2, 3]` | `[i64]` (heap) |
| `[for i in 0..5: i]` (constant bounds) | `[i64; 5]` |
| `[for i in 0..n: i]` (runtime bounds) | `[i64]` |

### Coercion

- `[T; N]` → `[T]`: implicit at function boundaries (zero-cost view — wraps pointer + known length)
- `[T]` → `[T; N]`: not allowed (size unknown at compile time)

### Index Types

- `isize` (signed) and `usize` (unsigned) map to MLIR `index` type (platform-native pointer width)
- Casting uses existing `as` keyword: `i as usize`, `n as isize`
- Allowed casts: integer ↔ `isize`/`usize`, float ↔ `isize`/`usize`. No bool conversions.
- Array indexing requires `usize` — `a[i]` where `i: i64` is a type error, user must write `a[i as usize]`

### Array Binding Semantics

- `let b = a` is a **reference** (alias) — passes the `memref` value, no copy
- Since polang currently has no mutation, copy vs alias is semantically equivalent
- Future: integrate MLIR's [Ownership-Based Buffer Deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/) to automatically manage copies when needed

### Built-in Functions

- `len(a)`: returns length for both `[T; N]` (compile-time constant) and `[T]` (runtime value)
- `len(a, dim)`: for multi-dimensional arrays, returns the size of dimension `dim` (0-indexed). `len(a)` is equivalent to `len(a, 0)`.
- Future: `shape(a)` will return a tuple of all dimension sizes when tuple types are available

## Mutability

### Design Principle

`mut` means **internal mutability** — the contents of a value can change, but the binding itself cannot be reassigned. `mut` always modifies the **value/type**, never the name. This model extends naturally to future structs where `mut` would indicate mutable fields.

### Declaration

```polang
let a = [1, 2, 3]           (* immutable array — elements cannot be modified *)
let a = mut [1, 2, 3]       (* mutable array — elements can be modified *)
def b = mut [1, 2, 3]       (* top-level mutable array *)
```

### Mutation with `<-`

The `<-` operator is used exclusively for mutating contents. It is never used for rebinding.

```polang
let a = mut [1, 2, 3]
a[0] <- 99                  (* OK — element mutation *)
```

### Errors

```polang
let a = mut 5               (* ERROR — mut on scalar, no inner state *)
a <- [4, 5, 6]              (* ERROR — rebinding not allowed *)
```

### Rules

- `mut` is only valid on types with inner state (arrays, slices, future structs)
- `<-` is only for content mutation (element assignment, future field assignment), never rebinding
- `=` is only used in declarations

## Function Parameters with Mutation

### Syntax

The `;` character separates regular parameters from mutable parameters inside the parameter list. (`|` is reserved for future algebraic data type variant syntax.)

```polang
(* regular params only *)
def add(x: i64, y: i64): i64 = x + y
add(1, 2)

(* mut params only *)
def clear(; arr: mut [i64]) =
  arr[0] <- 0
clear(; a)

(* mixed *)
def fill(val: i64; arr: mut [i64]) =
  arr[0] <- val
fill(99; a)

(* multiple mut params *)
def transfer(amount: i64; from: mut [i64], to: mut [i64]) = ...
transfer(100; src, dst)
```

### `mut` Placement

`mut` is always on the type side, consistent with declarations:

| Context | Syntax |
|---------|--------|
| Local binding | `let a = mut [1, 2, 3]` |
| Top-level | `def a = mut [1, 2, 3]` |
| Function param | `arr: mut [i64]` |

### Rules

- `;` separates regular params from mut params inside `()`
- Caller's variable must have been declared with `mut`
- Passing a non-mut array to a mut param is an error
- `[T; N]` implicitly coerces to `[T]` in mut position (view into stack memory)
- Element mutation allowed on mut params: `arr[i] <- expr`
- Whole reassignment **not** allowed on mut params: `arr <- expr` is an error
- This restriction exists because the caller may have passed a fixed array coerced to a slice — replacing the whole array could write beyond allocated memory

## Complete Syntax Examples

```polang
(* === Arrays === *)
let a = [1, 2, 3]                       (* [i64; 3] — fixed, stack *)
let b: [i64] = [1, 2, 3]               (* [i64] — slice, heap *)
let c = [for i in 0..5: i * i]         (* [i64; 5] — fixed, constant bounds *)
let d = [for i in 0..n: i * i]         (* [i64] — slice, runtime bounds *)

(* === Element access === *)
a[0]                                     (* 1 *)
len(a)                                   (* 3 *)

(* === Mutability === *)
let e = mut [1, 2, 3]                   (* mutable array *)
e[0] <- 99                              (* OK *)

let f = [1, 2, 3]                       (* immutable *)
f[0] <- 99                              (* ERROR *)

(* === Functions with mut params === *)
def fill(val: i64; arr: mut [i64]) =
  arr[0] <- val

let g = mut [0, 0, 0]
fill(99; g)                             (* g[0] is now 99 *)

(* === Reduce comprehension === *)
let total = reduce acc = 0 for i in 0..5: acc + i * i    (* 30 *)

(* === Coercion === *)
def first(arr: [i64]): i64 = arr[0]
let h = [10, 20, 30]                    (* [i64; 3] *)
first(h)                                (* OK — coerced to [i64] *)
```

## MLIR Lowering

### Array and Slice Operations

| Polang | MLIR |
|--------|------|
| `[1, 2, 3]` (fixed) | `memref.alloca` + `memref.store` per element |
| `[1, 2, 3]` (slice) | `memref.alloc` + `memref.store` per element |
| `a[i]` (read) | `memref.load` (or `affine.load` in Phase 5) |
| `a[i] <- expr` | `memref.store` (or `affine.store` in Phase 5) |
| `[for i in 0..N: expr]` | `memref.alloca/alloc` + `scf.for` + `memref.store` |
| `reduce acc = init for i in 0..N: body` | `scf.for` with iter_arg accumulator |
| `len(a)` on `[T; N]` | constant `N` |
| `len(a)` on `[T]` | `memref.dim` |
| `[T; N]` → `[T]` coercion | `memref.cast` static to dynamic |

### Mut Parameter Lowering

Mut parameters are passed as `memref` references. Since `memref` is already a reference type in MLIR, no special handling is needed — the callee operates directly on the caller's memory. Non-mut array parameters can also use `memref` but the type checker prevents mutation at the source level.

## Future Considerations

### Ownership-Based Buffer Deallocation

MLIR's `ownership-based-buffer-deallocation` pass can be integrated later to automatically insert `memref.dealloc` for heap-allocated slices. Key facts from MLIR's design:

- Ownership (who frees) and mutation (who writes) are **orthogonal** — the deallocation pass doesn't care about mutability
- Function ABI: arguments are borrowed (caller retains ownership), return values transfer ownership to caller
- The pass tracks ownership via runtime `i1` values, not the type system

This means polang's mutability design does not conflict with future deallocation — they are independent concerns handled by different passes.

### Stack Escape Analysis

When `[T; N]` coerces to `[T]` in mut position, the resulting slice is a view into stack memory. This is safe for direct function calls (the callee's lifetime is bounded by the caller's). However, future features may introduce escape risks:

- Closures capturing a `mut [T]` parameter backed by stack memory
- Storing a coerced slice into a longer-lived data structure

The current MLIR verifier rejects functions returning array types, which prevents the simplest escape. Full escape analysis (or integration with MLIR's ownership-based buffer deallocation) will be needed when closures or data structures can hold array references.

### Growable Containers

Slices are fixed-size-at-creation. A future `Vec` type (requiring structs) will handle growable arrays. Until then, the functional pattern works:

```polang
def push(arr: [i64], val: i64): [i64] = ...
let a = mut [1, 2, 3]
let b = push(a, 4)          (* returns new slice *)
```

### ADT Mutability

When ADTs are added, `mut` will allow inner field mutation but not variant switching:

```polang
(* Product type (struct) — single variant *)
let p = mut Point(1, 2)
p.x <- 10                   (* OK — field mutation *)

(* Sum type — multiple variants *)
let x = mut Some(5)
x.0 <- 10                   (* OK — field mutation within current variant *)
x <- None                   (* ERROR — variant switching is rebinding, not allowed *)

let y = mut None             (* ERROR — no inner state, like mut 5 *)
```

Variant switching requires functional construction via `match`. See [`docs/plans/2026-02-11-adt-compatibility-design.md`](2026-02-11-adt-compatibility-design.md) for full design decisions.

## Error Message Convention

### Format

```
<Category>: <description> at line <N>, column <M>
```

### Categories

| Category | Source | Scope |
|----------|--------|-------|
| `Syntax error` | Lexer/Parser | Malformed syntax: unterminated comments, missing brackets, unexpected tokens |
| `Type error` | Type checker | Type mismatches, undeclared variables, non-`usize` index, non-constant bounds, mixed array element types |

Only two categories, no error codes. Matches the existing convention (e.g., `Type error: Undeclared variable: x at line 5, column 1`). Each phase defines its specific error messages during the test-first workflow.

## Array Polymorphism

### Generic Type Infrastructure

Built-in structures get **dedicated MLIR types** for optimization. All share a common `PolangParametricTypeInterface` exposing `getTypeParams()` and `getName()`.

| Category | MLIR Type | Syntax | Example |
|----------|-----------|--------|---------|
| Array | `Polang_ArrayType` (dedicated) | `[i64; 5]` → `Array<i64, 5>` | `!polang.array<i64, 5>` |
| Tuple (future) | `Polang_TupleType` (dedicated) | TBD | `!polang.tuple<i64, f64>` |
| User-defined (future) | `Polang_ADTType` (unified) | TBD | `!polang.adt<"Option", [("None", []), ("Some", [i64])]>` |
| Box (future) | `Polang_BoxType` | TBD | `!polang.box<!polang.adt<"List">>` |

> **Note**: `Polang_ADTType` replaces the previously planned `Polang_StructType`. A struct is a single-variant ADT (e.g., `type Point = Point(i64, i64)`). Sum types have multiple variants (e.g., `type Option = None | Some(i64)`). See [`docs/plans/2026-02-11-adt-compatibility-design.md`](2026-02-11-adt-compatibility-design.md) for rationale.

### Polymorphism Levels

**Level 1** — Arrays as polymorphic arguments:
```polang
let identity(x) = x
identity([1, 2, 3])   (* monomorphizes to identity$array_i64_3 *)
```

**Level 2** — Comprehension bodies call polymorphic functions:
```polang
let double(x) = x + x
[for i in 0..5: double(i)]   (* works naturally *)
```

**Level 3** — Polymorphic over array element type:
```polang
let first(arr: ['a; 5]): 'a = arr[0]
first([1, 2, 3, 4, 5])           (* returns i64 *)
first([1.0, 2.0, 3.0, 4.0, 5.0]) (* returns f64 *)
```

## Implementation Phases

### Phase 0a: Comment Syntax Change

Replace `;` comments with `(* ... *)` OCaml-style block comments.

- Lexer: add `COMMENT` exclusive start condition with nesting counter
- Update lit test config: wrap directives in block comments (`(* RUN: ... *)`, `(* CHECK: ... *)`)
- Update all `.po` files and documentation

### Phase 0b: `isize`/`usize` Types

- `isize`, `usize` keywords in lexer
- New type names in parser `type_spec`
- Type checker: new types, extend cast validation
- MLIR: map to `index` type, extend `as` cast lowering

### Phase 1: Type System Refactor + Array Types, Literals, and Slices

**Type representation refactor** (prerequisite for arrays and future ADTs):
- Replace string-based type representation (`std::string`) with a `PolangType` object hierarchy
- `PolangType` base class with subclasses: `PrimitiveType`, `ArrayType`, `TypeVarType`, `UnificationVarType`, `BoxType` (stub for future ADTs), `FunctionType` (stub)
- Built-in types as pre-registered singleton instances (e.g., `PrimitiveType::I64`)
- Refactor type checker to use `shared_ptr<const PolangType>` instead of `std::string`
- Refactor `MonoSignature`/`PolymorphicSignature` to use `PolangType`
- Refactor unification and substitution to operate on `PolangType` objects
- Update `PolangTypeConverter` to convert from `PolangType` objects
- Design extensible for future user-defined types (ADTs — see [`docs/plans/2026-02-11-adt-compatibility-design.md`](2026-02-11-adt-compatibility-design.md))
- Include `BoxType` stub (heap-allocated pointer type, for future recursive ADTs — no MLIR ops or lowering yet)

**Array types and literals:**
- Fixed array type `[T; N]` and slice type `[T]`
- Array literal syntax `[expr, expr, ...]`
- `Polang_ArrayType` in MLIR dialect (parameterized by element type and optional static size)
- `PolangParametricTypeInterface` for generic type infrastructure
- Stack allocation for fixed arrays, heap allocation for slices
- Type inference: literals → `[T; N]`, explicit annotation for `[T]`
- Structural unification in type checker, monomorphization support for arrays
- `len()` built-in function
- MLIR verifier: reject functions returning array types (stack escape prevention)

### Phase 2: Element Access and Mutability

- `a[i]` element access syntax — index must be `usize`
- `mut` keyword in lexer
- `let a = mut [...]` declaration syntax
- `<-` operator for element mutation
- Type checker: `<-` only on `mut` values, only on types with inner state
- Type checker: structural type inference for arrays (Level 3 polymorphism)
- `mut` on scalar is a type error
- Compile-time bounds checking for constant indices

### Phase 3: Array Comprehensions

- `[for i in lo..hi: body]` comprehension syntax
- `for`, `in`, `..` tokens in lexer
- Loop variable type inferred from usage in body (default `i64` if unconstrained; `usize` if used as array index; type error if conflicting constraints)
- Runtime bounds → slice type inference
- Constant bounds (integer literals only) → fixed array type inference
- Future: constant-fold `len(a)` on `[T; N]` to treat as compile-time constant bound
- Loop variable scoped to body, can shadow outer variables

### Phase 4a: Reduce Comprehensions

- `reduce acc = init for i in lo..hi: body` syntax producing a scalar
- `reduce` keyword in lexer
- Body parsed greedily (low precedence, right-associative)
- `scf.for` with iter_arg accumulation

### Phase 4b: Function Mut Parameters

- `;` separator in parameter lists
- `arr: mut [T]` parameter syntax
- Caller must pass `mut` value
- Implicit `[T; N]` → `[T]` coercion at function boundaries
- Element mutation allowed, whole reassignment not allowed on mut params

### Phase 5: Affine Lowering Pass

- New `PolangToAffine.cpp` pass
- Array ops → `affine.for` + `affine.load`/`affine.store`
- `polang.reduce` → `affine.for` with iter_arg
- Non-affine indices: `matchAndRewrite` returns `failure()`, PolangToStandard handles fallback
- `index` type is builtin; `arith.index_cast` suffices (no IndexDialect needed)

### Phase 6: Affine Optimization Pass Integration

- `--affine-opt=<N>` flag (0=off, 1=basic, 2=aggressive)
- Level 1: loop fusion, scalar replacement, LICM
- Verify pass names against LLVM 20 headers

### Phase 7: Advanced Optimizations

- Level 2: loop tiling, parallelization, unrolling
- `--tile-size=<N>` CLI flag (default 32)

### Phase 7b: Multi-Dimensional Arrays

- `[i64; (3, 5)]` syntax for 2D arrays → `memref<3x5xi64>`
- `a[i, j]` multi-index access syntax
- Multi-index comprehension: `[for i in 0..3, j in 0..5: expr]` → `[T; (3, 5)]`
- Multi-index reduce: `reduce acc = 0 for i in 0..3, j in 0..5: body`
- `len(a)` returns first dimension, `len(a, N)` returns Nth dimension
- Future: `shape(a)` returns tuple when tuple types are available
- Parser, AST, type checker, and MLIR lowering for N-dimensional arrays
- Extends `Polang_ArrayType` with multi-dimensional size parameters

### Phase 8: Documentation, Examples, and Polish

- Update `doc/Syntax.md`, `doc/TypeSystem.md`, `doc/Architecture.md`
- New examples: `arrays.po`, `dot_product.po`, `matrix.po`
- Array display formatting in REPL

## Phase Dependencies

```
Phase 0a (Comment Syntax)
    |
Phase 0b (isize/usize Types)
    |
Phase 1 (Array Types + Literals + Slices)
    |
Phase 2 (Element Access + Mutability)
    |
    +--------------------------+
    |                          |
Phase 3 (Comprehensions)   Phase 4b (Mut Parameters)
    |                          |
Phase 4a (Reduce)              |
    |                          |
    +--------------------------+
    |
Phase 5 (Affine Lowering)
    |
Phase 6 (Affine Optimization)
    |
Phase 7 (Tiling + Parallelization)
    |
Phase 7b (Multi-Dimensional Arrays)
    |
Phase 8 (Documentation + Polish)
```

## Verification Strategy

After each phase:

1. Build: `cmake --build --preset clang-debug`
2. Test: `ctest --preset clang-debug`
3. Format: `./scripts/run-clang-format.sh`
4. Examples: `for f in example/*.po; do ./build/bin/PolangRepl "$f"; done`
