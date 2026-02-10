# ADT Compatibility Design Decisions

## Summary

This document records design decisions made to ensure future algebraic data types (ADTs) are compatible with the current feature roadmap (issue #33: arrays, slices, mutability, affine pipeline). These decisions were made proactively to avoid conflicts and minimize rework when ADTs are implemented.

## Context

Polang's roadmap includes arrays, slices, comprehensions, mutability, and polyhedral optimization (issue #33, Phases 0–8). ADTs (sum types + product types) are a future goal. This analysis identified 5 potential conflict areas and resolved each.

## Decision 1: `mut` Semantics With Sum Types

**Conflict**: The `<-` operator is defined as "content mutation only, never rebinding." For sum types, switching variants (e.g., `Some(5)` → `None`) changes the entire representation — is that content mutation or rebinding?

**Decision**: `mut` on ADTs allows **inner field mutation only**, not variant switching.

| Operation | Allowed? | Reason |
|-----------|----------|--------|
| `x.0 <- 10` (mutate field within current variant) | Yes | Content mutation |
| `x <- Some(10)` (replace with same variant) | No | Whole-value replacement = rebinding |
| `x <- None` (switch to different variant) | No | Variant switching = rebinding |

**Rules**:
- `mut` on a nullary constructor (e.g., `mut None`) is an error — no inner state, like `mut 5`
- `mut` on a constructor with fields (e.g., `mut Some(5)`) allows field mutation via `<-`
- Variant switching requires functional construction via `match`:
  ```polang
  let y = match x with
  | Some(v) -> None
  | None -> Some(0)
  ```

**Impact on current plan**: None. The current `mut`/`<-` design extends naturally to ADTs.

## Decision 2: Memory Model for Recursive ADTs

**Conflict**: The current plan uses `memref` for arrays (contiguous memory). Recursive ADTs like `type List = Nil | Cons(i64, List)` need heap-allocated pointer-linked nodes, which `memref` cannot represent.

**Decision**: Plan for `polang.box<T>` now, implement later.

- Add a `BoxType` stub to the Phase 1 `PolangType` hierarchy (type representation only, no MLIR ops or lowering)
- `polang.box<T>` will represent a heap-allocated pointer to `T`
- Recursive ADT fields use `Box<T>` for indirection: `Cons(i64, Box<List>)`
- Deallocation strategy (reference counting, tracing GC, or ownership) deferred to ADT implementation time

**MLIR representation (future)**:
```
!polang.box<!polang.adt<"List">>
```

**Impact on current plan**: Phase 1 adds `BoxType` as a subclass in the `PolangType` hierarchy (stub only, not implemented).

## Decision 3: Unified MLIR Type for User-Defined Types

**Conflict**: The plan includes `Polang_StructType` for user-defined types, but this only models product types (structs). ADTs are sum types (tagged unions of multiple variants).

**Decision**: Replace `Polang_StructType` with a unified `Polang_ADTType` that represents both products and sums.

A struct is a single-variant ADT:
```
type Point = Point(i64, i64)       (* 1 variant → product type / struct *)
type Option = None | Some(i64)     (* 2 variants → sum type *)
```

**Updated MLIR type table**:

| Category | MLIR Type | Example |
|----------|-----------|---------|
| Array | `Polang_ArrayType` (dedicated) | `!polang.array<i64, 5>` |
| Tuple (future) | `Polang_TupleType` (dedicated) | `!polang.tuple<i64, f64>` |
| User-defined (future) | `Polang_ADTType` (unified) | `!polang.adt<"Option", [("None", []), ("Some", [i64])]>` |
| Box (future) | `Polang_BoxType` | `!polang.box<!polang.adt<"List">>` |

**Lowering optimization**:
- 1 variant (struct) → no tag field, just payload
- 2+ variants (sum) → tag + union layout (or tagged pointer for small payloads)

**Impact on current plan**: Rename `Polang_StructType` → `Polang_ADTType` in design docs and issue #33.

## Decision 4: Constructor Syntax — Naming Convention

**Conflict**: ADT constructors (e.g., `Some(42)`) look syntactically identical to function calls (e.g., `add(42)`). How does the parser distinguish them?

**Decision**: Uppercase naming convention with three tiers:

| Category | Convention | Examples |
|----------|-----------|---------|
| Constructors | PascalCase | `Some`, `None`, `Red`, `Cons`, `Nil` |
| Constants | SCREAMING_SNAKE_CASE | `PI`, `MAX_SIZE`, `E` |
| Functions/variables | lowerCamelCase | `add`, `myFunc`, `x` |

**Rules**:
- Constructors must start with an uppercase letter and contain at least one lowercase letter
- Constants are all-uppercase (optionally with underscores)
- Functions and variables start with a lowercase letter
- Module names (PascalCase) are disambiguated by `.` (dot access): `Math.add` vs `Some(42)`
- Enforced in the parser when ADTs are implemented

**Impact on current plan**: None. Convention enforced only when ADTs are added.

## Decision 5: `let` Destructuring — Irrefutable Patterns Only

**Conflict**: `let` bindings are currently infallible. ADT destructuring can fail (e.g., `let Some(x) = expr` when `expr` is `None`).

**Decision**: `let` only supports irrefutable patterns (patterns that always match). Refutable patterns require `match`.

| Pattern | Allowed in `let`? | Reason |
|---------|-------------------|--------|
| `let Point(x, y) = p` | Yes | Single-variant ADT, always matches |
| `let Some(x) = expr` | **No** | Multi-variant, could be `None` — use `match` |
| `let (a, b) = tuple` | Yes | Tuples are single-variant |

**Match is required for sum types**:
```polang
match expr with
| Some(x) -> x + 1
| None -> 0
```

**Impact on current plan**: None. Additive feature when ADTs arrive.

## Summary Table

| # | Conflict | Decision | Impact on Current Plan |
|---|----------|----------|----------------------|
| 1 | `mut` vs sum types | Inner field mutation only, no variant switching | None |
| 2 | Memory model for recursive ADTs | Plan `polang.box<T>` now, implement later | Phase 1: add `BoxType` stub |
| 3 | MLIR type for user-defined types | Unified `Polang_ADTType` (replaces `Polang_StructType`) | Rename in design docs |
| 4 | Constructor vs function call syntax | PascalCase convention for constructors | None now |
| 5 | `let` destructuring | Irrefutable patterns only | None now |

## Future Work

When ADTs are implemented, the following will be needed (tracked in future issues):
- `type` keyword, ADT declarations, constructors, `match` expression — #56
- `Polang_BoxType` implementation, heap allocation/deallocation strategy — #57
- Pattern matching with exhaustiveness checking, `let` irrefutable destructuring — #58
- PascalCase constructor / SCREAMING_SNAKE constant naming enforcement — #59
