# Reject Mutual Recursion in Let-Groups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

> **Note:** The map consolidation PR (#48) has been merged. References to
> `functionReturnTypes`, `functionParamTypes`, and `functionSchemes` in this plan
> are now a single `functionSignatures` map of type
> `std::map<std::string, polang::FunctionSignature>` where `FunctionSignature` is
> `std::variant<MonoSignature, PolymorphicSignature>`. Adjust the implementation
> steps accordingly.

**Goal:** Make the type checker reject mutual recursion between sibling functions in let-groups, matching the existing behavior where sibling variables cannot reference each other.

**Architecture:** The root cause is that `visit(NMethodCall)` silently accepts calls to functions not in `functionSignatures` — it skips validation and defaults the return type to `i64`. The fix adds an "undefined function" error when a call references a name not found in `functionSignatures` or `localTypes`. This makes `is_even` calling `is_odd` (before `is_odd` is registered) produce a type error, just like `y = x + 1` already fails for sibling variables.

**Tech Stack:** C++, LLVM lit test infrastructure

---

### Task 1: Write the lit test for rejected mutual recursion

**Files:**
- Create: `tests/lit/Errors/mutual-recursion.po`

**Step 1: Write the test file**

```polang
; RUN: %not %polang_compiler %s 2>&1 | %FileCheck %s

; Mutual recursion between let-group siblings should be rejected.
; The first function (is_even) calls is_odd, which hasn't been declared yet.
; CHECK: Type error: Undefined function: is_odd
let is_even(n: i64): i64 = if n == 0 then 1 else is_odd(n - 1) and is_odd(n: i64): i64 = if n == 0 then 0 else is_even(n - 1) in is_even(4)
```

**Step 2: Run the test to verify it fails (the error is NOT yet produced)**

Run: `docker exec polang bash -c 'cd /workspace/polang && ./build/bin/llvm-lit -v tests/lit/Errors/mutual-recursion.po'`
Expected: FAIL — currently mutual recursion succeeds, so the CHECK pattern won't match.

**Step 3: Commit**

```bash
git add tests/lit/Errors/mutual-recursion.po
git commit -m "test: add lit test for rejecting mutual recursion in let-groups"
```

---

### Task 2: Add `formatUndefinedFunc` to error_reporter.hpp

**Files:**
- Modify: `parser/include/parser/error_reporter.hpp:119-121`

**Step 1: Add the new error formatter after `formatUndeclaredVar`**

In `parser/include/parser/error_reporter.hpp`, after the `formatUndeclaredVar` function (line 121), add:

```cpp
/// Format an undefined function error message.
/// @param funcName The function name
[[nodiscard]] inline std::string
formatUndefinedFunc(const std::string& funcName) {
  return "Undefined function: " + funcName;
}
```

**Step 2: Verify no compilation errors**

Run: `docker exec polang bash -c 'cd /workspace/polang && clangd --compile-commands-dir=build/clang-debug --check=/workspace/polang/parser/src/type_checker.cpp 2>&1 | grep -E "error|warning"'`
Expected: No new errors.

**Step 3: Commit**

```bash
git add parser/include/parser/error_reporter.hpp
git commit -m "feat: add formatUndefinedFunc error message helper"
```

---

### Task 3: Report error for undefined function calls in type checker

**Files:**
- Modify: `parser/src/type_checker.cpp:259-328` (the `visit(NMethodCall&)` method)

**Step 1: Add the using declaration**

At the top of `type_checker.cpp`, near line 22 where other `using` declarations are, add:

```cpp
using polang::formatUndefinedFunc;
```

**Step 2: Add undefined function error in `visit(NMethodCall&)`**

In the `visit(NMethodCall&)` method, after the `functionSchemes` check (line 274) and the `functionParamTypes` check (line 276-321), at the fallthrough point where neither map contains the function, add an error report. Specifically, change lines 322-328 from:

```cpp
  const auto funcReturnIt = functionReturnTypes.find(funcName);
  if (funcReturnIt != functionReturnTypes.end()) {
    inferredType = funcReturnIt->second;
  } else {
    inferredType = TypeNames::I64;
  }
```

to:

```cpp
  const auto funcReturnIt = functionReturnTypes.find(funcName);
  if (funcReturnIt != functionReturnTypes.end()) {
    inferredType = funcReturnIt->second;
  } else if (functionParamTypes.find(funcName) == functionParamTypes.end() &&
             functionSchemes.find(funcName) == functionSchemes.end()) {
    reportError(formatUndefinedFunc(funcName), node.loc);
    inferredType = TypeNames::UNKNOWN;
  } else {
    inferredType = TypeNames::I64;
  }
```

This reports an error only when the function name is truly unknown (not in `functionParamTypes` AND not in `functionSchemes`). If it IS in `functionParamTypes` but not in `functionReturnTypes`, we keep the `i64` default (self-recursion with inferred return type).

**Step 3: Verify no compilation errors**

Run: `docker exec polang bash -c 'cd /workspace/polang && cmake --build build/clang-debug -j$(nproc) 2>&1 | tail -5'`
Expected: Build succeeds.

**Step 4: Commit**

```bash
git add parser/src/type_checker.cpp
git commit -m "feat: reject calls to undefined functions in type checker"
```

---

### Task 4: Run all tests and fix the CHECK pattern if needed

**Step 1: Run the new mutual recursion lit test**

Run: `docker exec polang bash -c 'cd /workspace/polang && ./build/bin/llvm-lit -v tests/lit/Errors/mutual-recursion.po'`
Expected: PASS — the type checker now rejects the mutual recursion.

**Step 2: Run all lit tests to check for regressions**

Run: `docker exec polang bash -c 'cd /workspace/polang && ./build/bin/llvm-lit -v tests/lit/ 2>&1 | tail -30'`
Expected: All tests pass. If any test relied on calling a function before it was declared (forward reference), it will fail here and needs investigation.

**Step 3: Run C++ unit tests**

Run: `docker exec polang bash -c 'cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure -E ReplIntegration 2>&1 | tail -20'`
Expected: All pass (excluding pre-existing ReplIntegration failures).

**Step 4: If the CHECK pattern in the test doesn't match the actual error message, update it**

The exact error message format depends on how `reportError` prepends context. Read the actual stderr output:

Run: `docker exec polang bash -c '/workspace/polang/build/bin/PolangCompiler /workspace/polang/tests/lit/Errors/mutual-recursion.po 2>&1'`

Adjust the `CHECK` line in `tests/lit/Errors/mutual-recursion.po` to match the actual output.

**Step 5: Commit**

```bash
git add -A
git commit -m "fix: adjust test expectations for mutual recursion error"
```

---

### Task 5: Run format check and final verification

**Step 1: Run clang-format**

Run: `docker exec polang bash -c 'cd /workspace/polang && ./scripts/run-clang-format.sh'`
Expected: No formatting issues (or auto-fixes applied).

**Step 2: Run full test suite one final time**

Run: `docker exec polang bash -c 'cd /workspace/polang && ctest --test-dir build/clang-debug --output-on-failure -E ReplIntegration 2>&1 | tail -10'`
Expected: All tests pass.

**Step 3: Commit any formatting fixes**

```bash
git add -A
git commit -m "chore: apply clang-format"
```
