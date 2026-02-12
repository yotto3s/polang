# Unit Type Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce `()` as a first-class unit type and require `() -> T` arrow syntax for zero-parameter function type signatures.

**Architecture:** Two new AST nodes (`NUnitType`, `NUnitLiteral`), parser grammar additions for `()` in type and expression positions, type checker updates to handle `NArrowType(NUnitType, T)` for zero-param functions, and unit erasure in MLIR codegen. Existing tests/examples are migrated to the new syntax.

**Tech Stack:** C++ (Bison parser, MLIR codegen), lit tests with FileCheck

**Branch:** `feature/haskell-style-definitions` (current branch, work here directly)

**Docker:** All build/test commands run inside Docker:
```bash
docker exec polang bash -c "cd /workspace/polang && <command>"
```

**Build command:**
```bash
docker exec polang bash -c "cd /workspace/polang && cmake --build build/clang-debug -j\$(nproc)"
```

**Test command:**
```bash
docker exec polang bash -c "cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure"
```

**Format command:**
```bash
docker exec polang bash -c "cd /workspace/polang && ./scripts/run-clang-format.sh"
```

---

### Task 1: Add NUnitType AST node

**Files:**
- Modify: `parser/include/parser/node.hpp` (after NForallType class, ~line 211)
- Modify: `parser/include/parser/visitor.hpp` (add forward decl + visit method)
- Modify: `parser/include/parser/ast_printer.hpp` (add visit method declaration)
- Modify: `parser/src/node.cpp` (add accept implementation)
- Modify: `parser/src/ast_printer.cpp` (add visit stub)
- Modify: `parser/src/type_checker.cpp` (add visit stub + FreeVariableCollector stub)
- Modify: `parser/include/parser/type_checker.hpp` (add visit declaration)
- Modify: `mlir/lib/MLIRGen/MLIRGen.cpp` (add visit stub)

**Step 1: Add NUnitType class to node.hpp**

After the `NForallType` class (~line 211), before the `makeTypeSpec` helper, add:

```cpp
// Unit type for type expressions: ()
class NUnitType : public NTypeSpec {
public:
  NUnitType() = default;
  [[nodiscard]] std::string getTypeName() const override { return "()"; }
  [[nodiscard]] std::unique_ptr<const NTypeSpec> clone() const override {
    auto copy = std::make_unique<NUnitType>();
    copy->loc = loc;
    return copy;
  }
  void accept(Visitor &visitor) const override;
};
```

**Step 2: Add NUnitType to visitor.hpp**

Add forward declaration at top (after `class NForallType;`):
```cpp
class NUnitType;
```

Add visit method in the `Type Specification Visitors` section (after `visit(const NForallType&)`):
```cpp
virtual void visit(const NUnitType& node) = 0;
```

**Step 3: Add accept implementation to node.cpp**

After the `NForallType::accept` line (~line 50), add:
```cpp
void NUnitType::accept(Visitor& visitor) const { visitor.visit(*this); }
```

**Step 4: Add visit stubs to all visitors**

In `parser/include/parser/ast_printer.hpp`, add after `visit(const NForallType&)`:
```cpp
void visit(const NUnitType& node) override;
```

In `parser/src/ast_printer.cpp`, add after `visit(const NForallType&)`:
```cpp
void ASTPrinter::visit(const NUnitType& /*node*/) {
  // Types are printed via getTypeName() in parent visitors
}
```

In `parser/include/parser/type_checker.hpp`, add after `visit(const NForallType&)`:
```cpp
void visit(const NUnitType& node) override;
```

In `parser/src/type_checker.cpp`, add visit stub after `visit(const NForallType&)` (~line 159):
```cpp
void TypeChecker::visit(const NUnitType& /*node*/) {
  // Unit type in type signatures — handled via applyFunctionSignature
}
```

Also in `FreeVariableCollector` class (~line 50), add:
```cpp
void visit(const NUnitType& /*node*/) override {}
```

In `mlir/lib/MLIRGen/MLIRGen.cpp`, add after `visit(const NForallType&)` (~line 109):
```cpp
void visit(const NUnitType& /*node*/) override {}
```

**Step 5: Build and verify**

```bash
docker exec polang bash -c "cd /workspace/polang && cmake --build build/clang-debug -j\$(nproc)"
```
Expected: Compiles successfully.

**Step 6: Run tests to verify no regressions**

```bash
docker exec polang bash -c "cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure"
```
Expected: Same pass count as before (no changes to behavior yet).

**Step 7: Format and commit**

```bash
docker exec polang bash -c "cd /workspace/polang && ./scripts/run-clang-format.sh"
git add parser/include/parser/node.hpp parser/include/parser/visitor.hpp parser/include/parser/ast_printer.hpp parser/include/parser/type_checker.hpp parser/src/node.cpp parser/src/ast_printer.cpp parser/src/type_checker.cpp mlir/lib/MLIRGen/MLIRGen.cpp
git commit -m "feat: add NUnitType AST node with visitor stubs"
```

---

### Task 2: Add NUnitLiteral AST node

**Files:**
- Modify: `parser/include/parser/node.hpp` (after NUnitType, before makeTypeSpec)
- Modify: `parser/include/parser/visitor.hpp` (add forward decl + visit method)
- Modify: `parser/include/parser/ast_printer.hpp` (add visit declaration)
- Modify: `parser/src/node.cpp` (add accept)
- Modify: `parser/src/ast_printer.cpp` (add visit)
- Modify: `parser/src/type_checker.cpp` (add visit + FreeVariableCollector)
- Modify: `parser/include/parser/type_checker.hpp` (add visit declaration)
- Modify: `mlir/lib/MLIRGen/MLIRGen.cpp` (add visit stub)

**Step 1: Add NUnitLiteral class to node.hpp**

After NUnitType, add:

```cpp
// Unit literal expression: ()
class NUnitLiteral : public NExpression {
public:
  NUnitLiteral() = default;
  void accept(Visitor &visitor) const override;
};
```

**Step 2: Add to visitor.hpp**

Forward declaration (after `class NUnitType;`):
```cpp
class NUnitLiteral;
```

Visit method in Expression Visitors section (after `visit(const NLetExpression&)`):
```cpp
virtual void visit(const NUnitLiteral& node) = 0;
```

**Step 3: Add accept to node.cpp**

After NUnitType::accept:
```cpp
void NUnitLiteral::accept(Visitor& visitor) const { visitor.visit(*this); }
```

**Step 4: Add visit to all visitors**

In `parser/include/parser/ast_printer.hpp`, add in Expression section:
```cpp
void visit(const NUnitLiteral& node) override;
```

In `parser/src/ast_printer.cpp`:
```cpp
void ASTPrinter::visit(const NUnitLiteral& /*node*/) {
  printPrefix();
  out << "NUnitLiteral ()\n";
}
```

In `parser/include/parser/type_checker.hpp`, add in Expression section:
```cpp
void visit(const NUnitLiteral& node) override;
```

In `parser/src/type_checker.cpp`:
```cpp
void TypeChecker::visit(const NUnitLiteral& /*node*/) {
  inferredType = "()";
}
```

In `FreeVariableCollector`:
```cpp
void visit(const NUnitLiteral& /*node*/) override {}
```

In `mlir/lib/MLIRGen/MLIRGen.cpp`:
```cpp
void visit(const NUnitLiteral& /*node*/) override {
  // Unit literal — no-op at MLIR level
}
```

**Step 5: Add `()` to polang_types.hpp**

In `TypeNames` struct, add:
```cpp
static constexpr const char* UNIT = "()";
```

In `parseTypeName()`, add before the final `return std::nullopt;`:
```cpp
if (name == TypeNames::UNIT || name == "()") {
  return TypeKind::Unknown;  // Unit has its own kind, but Unknown works for now
}
```

**Step 6: Build, test, format, commit**

```bash
docker exec polang bash -c "cd /workspace/polang && cmake --build build/clang-debug -j\$(nproc)"
docker exec polang bash -c "cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure"
docker exec polang bash -c "cd /workspace/polang && ./scripts/run-clang-format.sh"
git add parser/include/parser/node.hpp parser/include/parser/visitor.hpp parser/include/parser/ast_printer.hpp parser/include/parser/type_checker.hpp parser/src/node.cpp parser/src/ast_printer.cpp parser/src/type_checker.cpp mlir/lib/MLIRGen/MLIRGen.cpp parser/include/parser/polang_types.hpp
git commit -m "feat: add NUnitLiteral AST node and () type constant"
```

---

### Task 3: Add `()` to parser grammar

**Files:**
- Modify: `parser/src/parser.y`

**Step 1: Add forward declarations**

In the `%code requires` block (~line 15), add after `class NForallType;`:
```cpp
class NUnitType;
class NUnitLiteral;
```

**Step 2: Add `()` as unit type in type_atom rule**

In the `type_atom` rule (~line 432), change from:
```yacc
type_atom : ident {
              $$ = std::make_unique<const NNamedType>($1->name);
            }
          | TTYPEVAR {
              /* Type variable reference: 'a */
              $$ = std::make_unique<const NTypeVar>($1);
            }
          | TLPAREN type_expr TRPAREN {
              $$ = std::move($2);
            }
          ;
```

To:
```yacc
type_atom : ident {
              $$ = std::make_unique<const NNamedType>($1->name);
            }
          | TTYPEVAR {
              /* Type variable reference: 'a */
              $$ = std::make_unique<const NTypeVar>($1);
            }
          | TLPAREN TRPAREN {
              /* () unit type */
              $$ = std::make_unique<const NUnitType>();
            }
          | TLPAREN type_expr TRPAREN {
              $$ = std::move($2);
            }
          ;
```

**IMPORTANT:** The `TLPAREN TRPAREN` rule MUST come BEFORE the `TLPAREN type_expr TRPAREN` rule so Bison resolves the ambiguity correctly (empty parens = unit type, not empty type_expr).

**Step 3: Add `()` as unit literal in expr rule**

In the `expr` rule, find the `TLPAREN expr TRPAREN` rule (~line 521):
```yacc
     | TLPAREN expr TRPAREN { $$ = std::move($2); }
```

Add the unit literal rule BEFORE it:
```yacc
     | TLPAREN TRPAREN {
         $$ = std::make_unique<NUnitLiteral>();
         SET_LOC($$, @$);
       }
     | TLPAREN expr TRPAREN { $$ = std::move($2); }
```

**Step 4: Update %expect for shift/reduce conflicts**

The new `TLPAREN TRPAREN` rules may change the shift/reduce conflict count. After building, check if bison reports a different number and update the `%expect` directive (~line 197) accordingly. If the conflict count changes, adjust:
```yacc
%expect <new_count>
```

**Step 5: Build and verify**

```bash
docker exec polang bash -c "cd /workspace/polang && cmake --build build/clang-debug -j\$(nproc)"
```
Expected: Compiles. Check for bison warnings about shift/reduce conflicts.

**Step 6: Run tests**

```bash
docker exec polang bash -c "cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure"
```
Expected: Same pass count (no behavior change yet — `()` is parseable but not used in any tests).

**Step 7: Format and commit**

```bash
docker exec polang bash -c "cd /workspace/polang && ./scripts/run-clang-format.sh"
git add parser/src/parser.y
git commit -m "feat: add () unit type and literal to parser grammar"
```

---

### Task 4: Update type checker for unit type in function signatures

**Files:**
- Modify: `parser/src/type_checker.cpp` — `applyFunctionSignature()` method

**Step 1: Update applyFunctionSignature to handle NUnitType param**

In `applyFunctionSignature()` (~line 1378), find the section that checks for arrow type (~lines 1429-1439):

```cpp
  const auto* arrowType = dynamic_cast<const NArrowType*>(&innerSig.get());
  if (arrowType == nullptr) {
    if (node.arguments.empty()) {
      // Zero-param function: non-arrow signature is just the return type
      node.type = innerSig.get().clone();
      return;
    }
    reportError("type signature for '" + node.id->name +
                "' is not a function type");
    return;
  }
```

Replace with:

```cpp
  const auto* arrowType = dynamic_cast<const NArrowType*>(&innerSig.get());
  if (arrowType == nullptr) {
    if (node.arguments.empty()) {
      // Non-arrow type for zero-param function is no longer valid.
      // Must use () -> T syntax.
      reportError("type signature for '" + node.id->name +
                  "' must use () -> " + innerSig.get().getTypeName() +
                  " for zero-parameter functions");
      return;
    }
    reportError("type signature for '" + node.id->name +
                "' is not a function type");
    return;
  }
```

**Step 2: Handle NUnitType as param type in arrow signatures**

After the existing arrow type processing, find where `paramTypeRefs` is built (~lines 1442-1451):

```cpp
  std::vector<std::reference_wrapper<const NTypeSpec>> paramTypeRefs;
  const auto* productType =
      dynamic_cast<const NProductType*>(arrowType->paramType.get());
  if (productType != nullptr) {
    for (const auto& t : productType->types) {
      paramTypeRefs.push_back(std::cref(*t));
    }
  } else {
    paramTypeRefs.push_back(std::cref(*arrowType->paramType));
  }
```

Replace with:

```cpp
  // Check for unit type parameter: () -> T means zero-param function
  const auto* unitParam = dynamic_cast<const NUnitType*>(arrowType->paramType.get());
  if (unitParam != nullptr) {
    // () -> T: zero-param function
    if (!node.arguments.empty()) {
      reportError("type signature for '" + node.id->name +
                  "' has () parameter but definition has " +
                  std::to_string(node.arguments.size()) + " parameters");
      return;
    }
    node.type = arrowType->returnType->clone();
    return;
  }

  std::vector<std::reference_wrapper<const NTypeSpec>> paramTypeRefs;
  const auto* productType =
      dynamic_cast<const NProductType*>(arrowType->paramType.get());
  if (productType != nullptr) {
    for (const auto& t : productType->types) {
      paramTypeRefs.push_back(std::cref(*t));
    }
  } else {
    paramTypeRefs.push_back(std::cref(*arrowType->paramType));
  }
```

**Step 3: Update validateTypeNames for NUnitType**

In `validateTypeNames()` (~line 1471), add after the `NForallType` block (before the final `return true;`):

```cpp
  if (dynamic_cast<const NUnitType*>(&typeSpec) != nullptr) {
    return true;  // Unit type is always valid
  }
```

**Step 4: Build and test**

```bash
docker exec polang bash -c "cd /workspace/polang && cmake --build build/clang-debug -j\$(nproc)"
docker exec polang bash -c "cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure"
```
Expected: Some tests with old `f : i64` syntax for zero-param functions will now FAIL (this is expected — they'll be fixed in Task 5).

**Step 5: Format and commit**

```bash
docker exec polang bash -c "cd /workspace/polang && ./scripts/run-clang-format.sh"
git add parser/src/type_checker.cpp
git commit -m "feat: type checker handles () -> T for zero-param function signatures"
```

---

### Task 5: Migrate existing tests and examples to new syntax

**Files to update** (change `name : T` to `name : () -> T` for zero-param functions):

1. `tests/lit/Execution/closures.po` — line 15: `sum_all : i64` → `sum_all : () -> i64`
2. `tests/lit/MLIR/types.po` — line 13: `double_val : f64` → `double_val : () -> f64`, line 25: `bool_val : bool` → `bool_val : () -> bool`
3. `tests/lit/LLVMIR/bool-type.po` — line 11: `bool_true : bool` → `bool_true : () -> bool`, line 20: `bool_false : bool` → `bool_false : () -> bool`
4. `tests/lit/Errors/return-type-mismatch.po` — line 7: `f : i64` → `f : () -> i64`
5. `example/closures.po` — line 14: `sum_all : i64` → `sum_all : () -> i64`, also update the comment on line 10

**Step 1: Update each file**

For each file above, replace the bare return type annotation with the arrow syntax. The CHECK patterns in the lit tests should NOT change (they check MLIR/LLVM output, not source syntax). The only lit test content that changes is the Polang source lines.

Specific edits:

`tests/lit/Execution/closures.po` line 15:
```
sum_all : () -> i64
```

`tests/lit/MLIR/types.po` line 13:
```
double_val : () -> f64
```
Line 25:
```
bool_val : () -> bool
```

`tests/lit/LLVMIR/bool-type.po` line 11:
```
bool_true : () -> bool
```
Line 20:
```
bool_false : () -> bool
```

`tests/lit/Errors/return-type-mismatch.po` line 7:
```
f : () -> i64
```

`example/closures.po` line 10:
```
(* Zero-param functions use () -> return_type syntax *)
```
Line 14:
```
sum_all : () -> i64
```

**Step 2: Build and test**

```bash
docker exec polang bash -c "cd /workspace/polang && cmake --build build/clang-debug -j\$(nproc)"
docker exec polang bash -c "cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure"
```
Expected: ALL tests pass (including the migrated ones).

**Step 3: Verify examples**

```bash
docker exec polang bash -c "cd /workspace/polang && for f in example/*.po; do echo \"=== \$(basename \$f) ===\"; ./build/clang-debug/bin/PolangRepl \"\$f\"; done"
```
Expected: `closures.po` outputs `21 : i64` (unchanged).

**Step 4: Format and commit**

```bash
docker exec polang bash -c "cd /workspace/polang && ./scripts/run-clang-format.sh"
git add tests/lit/Execution/closures.po tests/lit/MLIR/types.po tests/lit/LLVMIR/bool-type.po tests/lit/Errors/return-type-mismatch.po example/closures.po
git commit -m "refactor: migrate zero-param function signatures to () -> T syntax"
```

---

### Task 6: Add new unit type tests

**Files:**
- Create: `tests/lit/AST/unit-type-signature.po`
- Create: `tests/lit/Execution/unit-type.po`

**Step 1: Write AST test**

Create `tests/lit/AST/unit-type-signature.po`:
```polang
(* RUN: %polang_compiler --dump-ast %s | %FileCheck %s
*)

(* Test that () -> i64 type signature parses correctly *)
(* CHECK: NTypeSignature 'f' : () -> i64
*)
(* CHECK: NFunctionDeclaration 'f'
*)
f : () -> i64
f() = 42
```

**Step 2: Write execution test**

Create `tests/lit/Execution/unit-type.po`:
```polang
(* RUN: %polang_repl %s | %FileCheck %s
*)

(* Test zero-param function with () -> T type signature *)
(* CHECK: 42 : i64
*)
f : () -> i64
f() = 42
f()
```

**Step 3: Build and run tests**

```bash
docker exec polang bash -c "cd /workspace/polang && cmake --build build/clang-debug -j\$(nproc)"
docker exec polang bash -c "cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure"
```
Expected: New tests pass alongside all existing tests.

**Step 4: Commit**

```bash
git add tests/lit/AST/unit-type-signature.po tests/lit/Execution/unit-type.po
git commit -m "test: add unit type signature tests"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `doc/Syntax.md`
- Modify: `doc/TypeSystem.md`

**Step 1: Update Syntax.md**

Find the section about type signatures and update it to reflect the new syntax:
- Zero-param functions: `name : () -> return_type`
- Mention `()` as the unit type
- Update any examples showing `name : return_type` for zero-param functions

**Step 2: Update TypeSystem.md**

Add a section about the unit type:
- `()` is a type with one value `()`
- Used in function signatures: `() -> T` for zero-param functions
- `() -> ()` for side-effect functions

**Step 3: Commit**

```bash
git add doc/Syntax.md doc/TypeSystem.md
git commit -m "docs: update syntax and type system docs for unit type"
```

---

## Files Modified Summary

| File | Tasks | Change |
|------|-------|--------|
| `parser/include/parser/node.hpp` | 1, 2 | Add NUnitType, NUnitLiteral classes |
| `parser/include/parser/visitor.hpp` | 1, 2 | Add forward decls + visit methods |
| `parser/include/parser/ast_printer.hpp` | 1, 2 | Add visit declarations |
| `parser/include/parser/type_checker.hpp` | 1, 2 | Add visit declarations |
| `parser/include/parser/polang_types.hpp` | 2 | Add UNIT type constant |
| `parser/src/node.cpp` | 1, 2 | Add accept implementations |
| `parser/src/ast_printer.cpp` | 1, 2 | Add visit implementations |
| `parser/src/type_checker.cpp` | 1, 2, 4 | Add visit stubs + applyFunctionSignature changes |
| `parser/src/parser.y` | 3 | Add () grammar rules |
| `mlir/lib/MLIRGen/MLIRGen.cpp` | 1, 2 | Add visit stubs |
| `tests/lit/Execution/closures.po` | 5 | Migrate to () -> T |
| `tests/lit/MLIR/types.po` | 5 | Migrate to () -> T |
| `tests/lit/LLVMIR/bool-type.po` | 5 | Migrate to () -> T |
| `tests/lit/Errors/return-type-mismatch.po` | 5 | Migrate to () -> T |
| `example/closures.po` | 5 | Migrate to () -> T |
| `tests/lit/AST/unit-type-signature.po` | 6 | New test |
| `tests/lit/Execution/unit-type.po` | 6 | New test |
| `doc/Syntax.md` | 7 | Update syntax docs |
| `doc/TypeSystem.md` | 7 | Update type system docs |
