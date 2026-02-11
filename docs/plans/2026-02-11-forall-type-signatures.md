# Forall Type Signatures Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `forall` syntax for explicit polymorphic type signatures, type name validation, and a trait registry skeleton.

**Architecture:** Extend the lexer/parser to recognize `forall` quantifiers and type variables (`'a`, `'b`) in type signatures. The type checker creates `PolymorphicSignature` directly from declared type parameters rather than discovering them via inference. A `TraitRegistry` replaces hardcoded trait bound checks. The downstream pipeline (MLIRGen, monomorphization, lowering) is unchanged.

**Tech Stack:** C++ (Bison parser, Flex lexer), MLIR, LLVM

**Before → After syntax:**
```
(* BEFORE: polymorphism only via omitted signature *)
identity(x) = x
identity(42)

(* AFTER: explicit forall *)
identity : forall 'a. 'a -> 'a
identity(x) = x
identity(42)

(* Constrained *)
add : forall 'a:Numeric. 'a * 'a -> 'a
add(x, y) = x + y

(* Multiple type variables *)
first : forall 'a, 'b. 'a * 'b -> 'a
first(x, y) = x
```

---

## Phase 1: Write Tests

Write all tests first. Commit and pause for review before implementation.

### Task 1.1: Execution Tests (`tests/lit/Execution/`)

```polang
(* forall-basic.po: basic forall identity *)
identity : forall 'a. 'a -> 'a
identity(x) = x
identity(42)
(* CHECK: 42 : i64 *)

(* forall-numeric.po: constrained numeric *)
add : forall 'a:Numeric. 'a * 'a -> 'a
add(x, y) = x + y
add(1, 2)
(* CHECK: 3 : i64 *)

(* forall-numeric-float.po: constrained numeric with float *)
fadd : forall 'a:Numeric. 'a * 'a -> 'a
fadd(x, y) = x + y
fadd(1.5, 2.5)
(* CHECK: 4.0 *)

(* forall-multi-typevar.po: multiple type variables *)
first : forall 'a, 'b. 'a * 'b -> 'a
first(x, y) = x
first(42, true)
(* CHECK: 42 : i64 *)

(* forall-mixed-bounds.po: mixed bounds *)
numFirst : forall 'a:Numeric, 'b. 'a * 'b -> 'a
numFirst(x, y) = x + x
numFirst(5, true)
(* CHECK: 10 : i64 *)
```

### Task 1.2: Error Tests (`tests/lit/Errors/`)

```polang
(* error-unknown-type-in-sig.po *)
foo : blah -> blah
foo(x) = x
(* CHECK: unknown type 'blah' *)

(* error-unknown-trait-bound.po *)
bar : forall 'a:Stringable. 'a -> 'a
bar(x) = x
(* CHECK: unknown type class 'Stringable' *)

(* error-typevar-without-forall.po *)
baz : 'a -> 'a
baz(x) = x
(* CHECK: error *)

(* error-undeclared-typevar.po *)
qux : forall 'a. 'a -> 'b
qux(x) = x
(* CHECK: undeclared type variable *)

(* error-missing-numeric-bound.po *)
bad : forall 'a. 'a * 'a -> 'a
bad(x, y) = x + y
(* CHECK: requires Numeric *)

(* error-type-violates-bound.po *)
wrong : forall 'a:Integer. 'a -> 'a
wrong(x) = x
wrong(3.14)
(* CHECK: does not satisfy Integer *)
```

### Task 1.3: Warning Tests (`tests/lit/Errors/`)

```polang
(* warn-unused-typevar.po *)
unused : forall 'a, 'b. 'a -> 'a
unused(x) = x
unused(42)
(* CHECK: unused type variable *)
```

### Task 1.4: MLIR Output Tests (`tests/lit/MLIR/`)

Verify forall signatures produce the same `generic_func`/`instantiate` MLIR as inference-based polymorphism.

### Task 1.5: AST Dump Tests (`tests/lit/AST/`)

Verify `NForallType` and `NTypeVar` appear correctly in `--dump-ast` output.

### Task 1.6: Commit tests, pause for review

---

## Phase 2: Implementation

After tests are reviewed and approved, implement to make them pass. If any test needs modification during implementation, ask for approval first.

### Task 2.1: Lexer — New Tokens

Add to `parser/src/lexer.l`:
- **`TFORALL`** — keyword `forall`
- **`TTYPEVAR`** — pattern `'[a-z][a-zA-Z0-9_]*` (captures full string including quote)
- **`TDOT`** — literal `.`

### Task 2.2: AST Nodes

Add to `parser/include/parser/node.hpp`:

**`NTypeVar`** (subclass of `NTypeSpec`) — type variable reference in the type body.
- Fields: `name` (string, e.g. `"'a"`)
- `getTypeName()` returns `"typevar"` to match existing internal convention

**`NForallType`** (subclass of `NTypeSpec`) — wraps the quantified type.
- Fields:
  - `typeVars` — vector of declared vars, each with name and optional bound string
  - `innerType` — the actual type expression (typically `NArrowType`)
- `getTypeName()` delegates to `innerType`

### Task 2.3: Parser — Grammar Rules

Add to `parser/src/parser.y`:

```
type_expr : TFORALL type_var_list TDOT type_expr
          | type_product TARROW type_expr
          | type_product

type_var_list : type_var_decl
              | type_var_list TCOMMA type_var_decl

type_var_decl : TTYPEVAR
              | TTYPEVAR TCOLON ident

type_atom : TTYPEVAR
          | ident
          | TLPAREN type_expr TRPAREN
```

### Task 2.4: Trait Registry

Add to `parser/include/parser/type_inference.hpp` (or new `trait_registry.hpp`):

```cpp
struct TraitMethodSignature {
  std::string methodName;
  std::vector<std::string> paramTypes;       // ["'self", "'self"]
  std::string returnType;                    // "'self" or "bool"
};

struct TraitDefinition {
  std::string name;
  std::vector<TraitMethodSignature> methods;
  std::set<std::string> satisfyingTypes;
};

class TraitRegistry {
  std::map<std::string, TraitDefinition> traits;
  std::map<std::string, std::string> methodToTrait;
public:
  void registerTrait(TraitDefinition def);
  bool isKnownTrait(const std::string& name) const;
  bool satisfies(const std::string& type, const std::string& trait) const;
  std::optional<std::string> traitForMethod(const std::string& method) const;
};
```

Populate with built-in traits:

| Trait | Methods | Satisfying Types |
|-------|---------|-----------------|
| `Numeric` | `+`, `-`, `*`, `/` | all integer + float types |
| `Integer` | inherits Numeric | `i8`–`i64`, `u8`–`u64`, `isize`, `usize` |
| `Float` | inherits Numeric | `f32`, `f64` |

### Task 2.5: Type Checker — `applyFunctionSignature`

When signature is `NForallType`:
1. Extract declared type variables and bounds
2. Set `node.typeParams` directly (e.g., `["'a", "'b"]`)
3. Set `node.typeParamBounds` from declared bounds
4. Apply param types — `NTypeVar` → `NNamedType("typevar")`
5. Set `hasExplicitForall` flag on the node

### Task 2.6: Type Checker — Type Name Validation

In `applyFunctionSignature`, walk the type tree and check every `NNamedType` is:
1. A known base type (`i64`, `f64`, `bool`, etc.)
2. A declared type variable (in the `forall` clause)
3. The internal `typevar` marker

Error immediately on anything else.

### Task 2.7: Type Checker — Trait Bound Validation

- Validate bound names against `TraitRegistry::isKnownTrait()`
- At call sites, validate concrete types via `TraitRegistry::satisfies()`
- Replace hardcoded operator-to-trait mapping with `TraitRegistry::traitForMethod()`
- Verify body constraints are sufficient per declared bounds

### Task 2.8: Type Checker — `inferFunction` Adjustments

When `hasExplicitForall` is true:
- Skip "missing type signature" warning
- Use declared type param names from the node
- Build `PolymorphicSignature` from declared params/bounds

### Task 2.9: Build, run tests, fix issues

### Task 2.10: Format code (`run-clang-format.sh`)

---

## Phase 3: Documentation & Cleanup

### Task 3.1: Update `doc/Syntax.md`

Document `forall` syntax, type variables, and bounds.

### Task 3.2: Update `doc/TypeSystem.md`

Document explicit polymorphic signatures and the trait registry.

### Task 3.3: Final commit and push

---

## Design Reference

### Surface Syntax

```
name : forall 'a. 'a -> 'a
name : forall 'a:Numeric. 'a * 'a -> 'a
name : forall 'a, 'b. 'a * 'b -> 'a
name : forall 'a:Numeric, 'b. 'a * 'b -> 'a
```

### MLIRGen & Downstream — No Changes

By producing the same AST representation (`typeParams`, `typeParamBounds`, `typevar`-typed params) that inference currently produces, all of MLIRGen, monomorphization, and lowering work without modification.

| Component | Changes? |
|-----------|----------|
| Lexer | Yes |
| Parser | Yes |
| AST nodes | Yes |
| Type checker | Yes |
| MLIRGen | No |
| Monomorphization | No |
| Lowering passes | No |

### Files Touched

| File | Change |
|------|--------|
| `parser/src/lexer.l` | `TFORALL`, `TTYPEVAR`, `TDOT` tokens |
| `parser/src/parser.y` | `forall` rule, `type_var` in `type_atom` |
| `parser/include/parser/node.hpp` | `NTypeVar`, `NForallType` nodes |
| `parser/include/parser/type_inference.hpp` | `TraitRegistry`, `TraitDefinition`, `TraitMethodSignature` |
| `parser/src/type_checker.cpp` | `applyFunctionSignature`, validation, registry-based inference |
| `tests/lit/` | New tests across AST, MLIR, Execution, Errors |
| `doc/Syntax.md` | Document `forall` syntax |
| `doc/TypeSystem.md` | Document explicit polymorphic signatures |

### Not In Scope

- `trait` / `impl` declaration syntax
- Method dispatch
- Trait inheritance
- Higher-kinded types
