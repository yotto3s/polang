# Unit Type and Arrow Syntax for Zero-Parameter Functions

## Summary

Introduce `()` as a first-class unit type in Polang and require arrow syntax `() -> T` for zero-parameter function type signatures, replacing the current bare return type `f : T`.

## Motivation

Currently, zero-parameter function signatures are ambiguous with variable bindings:

```polang
(* Is this a variable or a zero-param function? *)
x : i64
```

With the arrow syntax, functions are always explicit:

```polang
x : i64              (* variable binding *)
f : () -> i64        (* zero-param function *)
```

A first-class unit type also enables:
- `() -> ()` for side-effect-only functions
- `()` as a value in polymorphic contexts
- Consistent type system (all functions have arrow types)

## Design

### Type System Semantics

- `()` is a type with exactly one value, also written `()`
- All function type signatures use arrow notation:
  - `f : () -> i64` (zero params)
  - `f : i64 -> i64` (one param)
  - `f : i64 * i64 -> i64` (multiple params)
- At MLIR/LLVM level, unit parameters are erased (zero-arg functions stay zero-arg)

### AST Nodes

Two new nodes:

```cpp
// Unit type in type expressions: ()
class NUnitType : public NTypeSpec {
  std::string getTypeName() const override { return "()"; }
};

// Unit literal in expressions: ()
class NUnitLiteral : public NExpression {
  // The sole value of type ()
};
```

### Parser Grammar

```yacc
(* Type atom — add unit type *)
type_atom : TIDENT
          | TTYPEVAR
          | TLPAREN TRPAREN          (* () unit type — NEW *)
          | TLPAREN type_expr TRPAREN
          ;

(* Expression — add unit literal *)
primary_expr : ...existing...
             | TLPAREN TRPAREN       (* () unit literal — NEW *)
             ;
```

No lexer changes needed — `(` and `)` are already tokens.

Disambiguation:
- `()` alone = unit literal/type
- `f()` = zero-arg function call (existing NMethodCall rule)

### Type Checker

1. **Unit type recognition** — `NUnitType` maps to internal unit type
2. **Zero-param signatures** — `() -> T` (NArrowType with NUnitType param) matches `f() = ...`
3. **Breaking change** — bare `f : T` is no longer valid for zero-param functions; must use `f : () -> T`
4. **Validation:**
   - `NArrowType(NUnitType, T)` + `f() = ...` (0 params) -> valid
   - `NArrowType(NUnitType, T)` + `f(x) = ...` (has params) -> error
   - Non-arrow `x : T` + `x() = ...` -> error: not a function type

### MLIR/Codegen

Unit is erased at codegen level:

| Polang | MLIR | LLVM IR |
|--------|------|---------|
| `() -> i64` | `() -> i64` (no args) | `i64 ()` |
| `() -> ()` | `() -> ()` (void) | `void ()` |
| `()` value | NoneType / erased | not materialized |
| `x : ()` | variable of NoneType | optimized away |

## Implementation Plan

### Phase 1: AST & Parser
- [ ] Add `NUnitType` to node.hpp (with clone())
- [ ] Add `NUnitLiteral` to node.hpp
- [ ] Update parser.y: `()` in type_atom and primary_expr
- [ ] Update AST printer for new nodes

### Phase 2: Type Checker
- [ ] Add unit type to internal type representation
- [ ] Update `applyFunctionSignature()` to handle `NArrowType(NUnitType, T)` for 0-param functions
- [ ] Remove support for bare return type on function signatures
- [ ] Add type inference for `()` literal
- [ ] Add unit type equality checks

### Phase 3: MLIR Codegen
- [ ] Handle unit type in MLIRGen (erase unit params)
- [ ] Handle unit return type (map to void)
- [ ] Handle unit literal expression (no-op)

### Phase 4: Tests & Migration
- [ ] Update all existing zero-param function signatures in tests
- [ ] Update examples (closures.po, etc.)
- [ ] Add new tests: AST/unit-type.po, MLIR/unit-type.po, Execution/unit-type.po
- [ ] Add error tests: Errors/unit-type-mismatch.po
- [ ] Update doc/Syntax.md and doc/TypeSystem.md

## Breaking Changes

- `f : T` for zero-param functions is no longer valid
- Must use `f : () -> T` instead
- Variable bindings `x : T` are unaffected
