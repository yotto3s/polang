// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
#include <memory>
#include <string>

// clang-format off
// Parser headers
#include "parser/node.hpp"
#include "parser.hpp"
#include "parser/parser_api.hpp"
#include "parser/type_checker.hpp"
// clang-format on

// ============== Helper Functions ==============

std::vector<TypeCheckError> checkTypes(const std::string& source) {
  auto block = polang_parse(source);
  if (!block) {
    return {TypeCheckError("Parse error")};
  }
  return polang_check_types(*block);
}

bool hasTypeError(const std::string& source) {
  return !checkTypes(source).empty();
}

bool hasNoTypeError(const std::string& source) {
  return checkTypes(source).empty();
}

// ============== Valid Type Tests ==============

TEST(TypeCheckTest, IntegerArithmetic) {
  EXPECT_TRUE(hasNoTypeError("let x = 1 + 2"));
  EXPECT_TRUE(hasNoTypeError("let x = 1 - 2"));
  EXPECT_TRUE(hasNoTypeError("let x = 1 * 2"));
  EXPECT_TRUE(hasNoTypeError("let x = 1 / 2"));
}

TEST(TypeCheckTest, DoubleArithmetic) {
  EXPECT_TRUE(hasNoTypeError("let x : f64 = 1.0 + 2.0"));
  EXPECT_TRUE(hasNoTypeError("let x : f64 = 1.0 - 2.0"));
  EXPECT_TRUE(hasNoTypeError("let x : f64 = 1.0 * 2.0"));
  EXPECT_TRUE(hasNoTypeError("let x : f64 = 1.0 / 2.0"));
}

TEST(TypeCheckTest, VariableUsage) {
  EXPECT_TRUE(hasNoTypeError("let x = 1\nlet y = x + 1"));
  EXPECT_TRUE(hasNoTypeError("let x : f64 = 1.0\nlet y : f64 = x + 2.0"));
}

TEST(TypeCheckTest, FunctionDeclaration) {
  EXPECT_TRUE(hasNoTypeError("let add(x: i64, y: i64): i64 = x + y"));
  EXPECT_TRUE(hasNoTypeError("let mul(a: f64, b: f64): f64 = a * b"));
}

TEST(TypeCheckTest, LetExpression) {
  EXPECT_TRUE(hasNoTypeError("let x = 1 in x + 1"));
  EXPECT_TRUE(hasNoTypeError("let x = 1 and y = 2 in x + y"));
}

TEST(TypeCheckTest, IfExpression) {
  // If condition must be bool (comparison or boolean literal)
  EXPECT_TRUE(hasNoTypeError("if true then 2 else 3"));
  EXPECT_TRUE(hasNoTypeError("if 1 == 1 then 2 else 3"));
  EXPECT_TRUE(hasNoTypeError("let x = if true then 2 else 3"));
}

TEST(TypeCheckTest, Comparison) {
  // Comparisons return bool, so variable must be declared as bool
  EXPECT_TRUE(hasNoTypeError("let x : bool = 1 < 2"));
  EXPECT_TRUE(hasNoTypeError("let x : bool = 1 == 2"));
  EXPECT_TRUE(hasNoTypeError(
      "let x : f64 = 1.0\nlet y : f64 = 2.0\nlet z : bool = x < y"));
  // Using comparison in if condition (returns bool)
  EXPECT_TRUE(hasNoTypeError("if 1 < 2 then 3 else 4"));
}

// ============== Type Error Tests ==============

TEST(TypeCheckTest, MixedArithmeticTypes) {
  EXPECT_TRUE(hasTypeError("let x = 1 + 2.0"));
  EXPECT_TRUE(hasTypeError("let x = 1.0 - 2"));
  EXPECT_TRUE(hasTypeError("let x = 1 * 2.0"));
  EXPECT_TRUE(hasTypeError("let x = 1.0 / 2"));
}

TEST(TypeCheckTest, MixedComparisonTypes) {
  EXPECT_TRUE(hasTypeError("let x = 1 < 2.0"));
  EXPECT_TRUE(hasTypeError("let x = 1.0 == 2"));
}

TEST(TypeCheckTest, VariableTypeMismatch) {
  EXPECT_TRUE(hasTypeError("let x : i64 = 1.0"));
  EXPECT_TRUE(hasTypeError("let x : f64 = 1"));
}

TEST(TypeCheckTest, FunctionReturnTypeMismatch) {
  EXPECT_TRUE(hasTypeError("let f(x: i64): f64 = x"));
  EXPECT_TRUE(hasTypeError("let f(x: f64): i64 = x"));
}

TEST(TypeCheckTest, IfBranchTypeMismatch) {
  EXPECT_TRUE(hasTypeError("if true then 2 else 3.0"));
  EXPECT_TRUE(hasTypeError("if true then 2.0 else 3"));
}

TEST(TypeCheckTest, IfConditionMustBeBool) {
  EXPECT_TRUE(hasTypeError("if 1 then 2 else 3"));
  EXPECT_TRUE(hasTypeError("if 1.0 then 2 else 3"));
}

TEST(TypeCheckTest, UndeclaredVariable) {
  EXPECT_TRUE(hasTypeError("x + 1"));
  EXPECT_TRUE(hasTypeError("let y = x"));
}

TEST(TypeCheckTest, LetExpressionTypeMismatch) {
  EXPECT_TRUE(hasTypeError("let x = 1 in x + 2.0"));
  EXPECT_TRUE(hasTypeError("let x : i64 = 1 and y : f64 = 2.0 in x + y"));
}

TEST(TypeCheckTest, AssignmentTypeMismatch) {
  EXPECT_TRUE(hasTypeError("let mut x = 1\nx <- 2.0"));
}

// ============== Mutability Tests ==============

TEST(TypeCheckTest, MutableVariableReassignment) {
  // Mutable variable can be reassigned
  EXPECT_TRUE(hasNoTypeError("let mut x = 1\nx <- 2\nx"));
  EXPECT_TRUE(hasNoTypeError("let mut x: i64 = 1\nx <- 2\nx"));
}

TEST(TypeCheckTest, ImmutableVariableReassignment) {
  // Immutable variable cannot be reassigned
  EXPECT_TRUE(hasTypeError("let x = 1\nx <- 2"));
  EXPECT_TRUE(hasTypeError("let x: i64 = 1\nx <- 2"));
}

TEST(TypeCheckTest, MutableInLetExpression) {
  // Mutable binding in let expression
  EXPECT_TRUE(hasNoTypeError("let mut x = 1 in x <- 2"));
  EXPECT_TRUE(hasNoTypeError("let mut x = 1 and y = 2 in x <- 10"));
}

TEST(TypeCheckTest, ImmutableInLetExpression) {
  // Immutable binding in let expression cannot be reassigned
  EXPECT_TRUE(hasTypeError("let x = 1 in x <- 2"));
  EXPECT_TRUE(hasTypeError("let x = 1 and mut y = 2 in x <- 10"));
}

TEST(TypeCheckTest, MultipleReassignments) {
  // Multiple reassignments of the same mutable variable
  EXPECT_TRUE(hasNoTypeError("let mut x = 1\nx <- 2\nx <- 3\nx"));
}

TEST(TypeCheckTest, MutableDoubleType) {
  // Mutable double variable
  EXPECT_TRUE(hasNoTypeError("let mut x: f64 = 1.0\nx <- 2.5\nx"));
  EXPECT_TRUE(hasTypeError("let mut x: f64 = 1.0\nx <- 2")); // int to double
}

TEST(TypeCheckTest, MutableBoolType) {
  // Mutable bool variable
  EXPECT_TRUE(hasNoTypeError("let mut flag: bool = true\nflag <- false\nflag"));
}

TEST(TypeCheckTest, MixedMutabilityInLetExpression) {
  // Mix of mutable and immutable bindings
  EXPECT_TRUE(hasNoTypeError("let x = 1 and mut y = 2 in y <- 10"));
  EXPECT_TRUE(hasNoTypeError("let mut a = 1 and b = 2 in a <- b"));
}

TEST(TypeCheckTest, NestedLetWithMutability) {
  // Nested let expressions with mutable bindings
  EXPECT_TRUE(hasNoTypeError("let mut x = 1 in let y = x in y"));
  EXPECT_TRUE(hasNoTypeError("let x = 1 in let mut y = x in y <- 2"));
}

TEST(TypeCheckTest, ImmutableReassignmentErrorMessage) {
  // Error message should contain the variable name
  auto errors = checkTypes("let x = 1\nx <- 2");
  ASSERT_FALSE(errors.empty());
  EXPECT_TRUE(errors[0].message.find("x") != std::string::npos);
  EXPECT_TRUE(errors[0].message.find("immutable") != std::string::npos);
}

// ============== Error Message Tests ==============

TEST(TypeCheckTest, ErrorMessageContainsOperator) {
  auto errors = checkTypes("let x = 1 + 2.0");
  ASSERT_FALSE(errors.empty());
  EXPECT_TRUE(errors[0].message.find("+") != std::string::npos);
}

TEST(TypeCheckTest, ErrorMessageContainsTypes) {
  auto errors = checkTypes("let x = 1 + 2.0");
  ASSERT_FALSE(errors.empty());
  EXPECT_TRUE(errors[0].message.find("i64") != std::string::npos);
  EXPECT_TRUE(errors[0].message.find("f64") != std::string::npos);
}

// ============== Type Inference Tests ==============

TEST(TypeCheckTest, InferIntFromLiteral) {
  // let x = 42 should infer int
  EXPECT_TRUE(hasNoTypeError("let x = 42\nlet y: i64 = x"));
}

TEST(TypeCheckTest, InferDoubleFromLiteral) {
  // let x = 3.14 should infer double
  EXPECT_TRUE(hasNoTypeError("let x = 3.14\nlet y: f64 = x"));
}

TEST(TypeCheckTest, InferBoolFromLiteral) {
  // let x = true should infer bool
  EXPECT_TRUE(hasNoTypeError("let x = true\nif x then 1 else 0"));
}

TEST(TypeCheckTest, InferFromExpression) {
  // let x = 1 + 2 should infer int
  EXPECT_TRUE(hasNoTypeError("let x = 1 + 2\nlet y: i64 = x"));
}

TEST(TypeCheckTest, InferFromComparison) {
  // let x = 1 < 2 should infer bool
  EXPECT_TRUE(hasNoTypeError("let x = 1 < 2\nif x then 1 else 0"));
}

TEST(TypeCheckTest, InferFunctionReturnType) {
  // let f(x: i64) = x + 1 should infer int return type
  EXPECT_TRUE(hasNoTypeError("let f(x: i64) = x + 1\nlet y: i64 = f(5)"));
}

TEST(TypeCheckTest, InferFunctionReturnTypeDouble) {
  // let f(x: f64) = x + 1.0 should infer double return type
  EXPECT_TRUE(
      hasNoTypeError("let f(x: f64) = x + 1.0\nlet y: f64 = f(5.0)"));
}

TEST(TypeCheckTest, NoImplicitConversionIntToDouble) {
  // let x: f64 = 42 should be error (no coercion)
  EXPECT_TRUE(hasTypeError("let x: f64 = 42"));
}

TEST(TypeCheckTest, NoImplicitConversionDoubleToInt) {
  // let x: i64 = 42.0 should be error (no coercion)
  EXPECT_TRUE(hasTypeError("let x: i64 = 42.0"));
}

TEST(TypeCheckTest, LetExpressionInferInt) {
  // let x = 1 in x + 1 should work (x inferred as int)
  EXPECT_TRUE(hasNoTypeError("let x = 1 in x + 1"));
}

TEST(TypeCheckTest, LetExpressionInferDouble) {
  // let x = 1.0 in x + 1.0 should work (x inferred as double)
  EXPECT_TRUE(hasNoTypeError("let x = 1.0 in x + 1.0"));
}

TEST(TypeCheckTest, LetExpressionInferMismatch) {
  // let x = 1 in x + 1.0 should fail (int + double)
  EXPECT_TRUE(hasTypeError("let x = 1 in x + 1.0"));
}

TEST(TypeCheckTest, InferredVariableUsedWithWrongType) {
  // let x = 42 followed by double operation should fail
  EXPECT_TRUE(hasTypeError("let x = 42\nlet y = x + 1.0"));
}

// ============== Function Call Type Checking Tests ==============

TEST(TypeCheckTest, FunctionCallCorrectTypes) {
  // Correct argument types should pass
  EXPECT_TRUE(hasNoTypeError("let f(x: i64) = x + 1\nf(5)"));
  EXPECT_TRUE(hasNoTypeError("let f(x: f64) = x + 1.0\nf(5.0)"));
  EXPECT_TRUE(hasNoTypeError("let f(x: i64, y: i64) = x + y\nf(1, 2)"));
}

TEST(TypeCheckTest, FunctionCallWrongArgType) {
  // Passing double to int parameter should fail
  EXPECT_TRUE(hasTypeError("let f(x: i64) = x + 1\nf(3.5)"));
  // Passing int to double parameter should fail
  EXPECT_TRUE(hasTypeError("let f(x: f64) = x + 1.0\nf(3)"));
}

TEST(TypeCheckTest, FunctionCallWrongArgCount) {
  // Too few arguments
  EXPECT_TRUE(hasTypeError("let f(x: i64, y: i64) = x + y\nf(1)"));
  // Too many arguments
  EXPECT_TRUE(hasTypeError("let f(x: i64) = x + 1\nf(1, 2)"));
}

TEST(TypeCheckTest, FunctionCallMultipleArgsTypeMismatch) {
  // Second argument has wrong type
  EXPECT_TRUE(hasTypeError("let f(x: i64, y: i64) = x + y\nf(1, 2.0)"));
  // First argument has wrong type
  EXPECT_TRUE(hasTypeError("let f(x: i64, y: i64) = x + y\nf(1.0, 2)"));
}

TEST(TypeCheckTest, FunctionCallErrorMessage) {
  auto errors = checkTypes("let f(x: i64) = x + 1\nf(3.5)");
  ASSERT_FALSE(errors.empty());
  EXPECT_TRUE(errors[0].message.find("i64") != std::string::npos);
  EXPECT_TRUE(errors[0].message.find("f64") != std::string::npos);
}

// ============== Function in Let Expression Type Checking Tests ==============

TEST(TypeCheckTest, LetExpressionWithFunction) {
  // Function declared in let expression should be type-checked
  EXPECT_TRUE(hasNoTypeError("let f(x: i64): i64 = x + 1 in f(5)"));
  EXPECT_TRUE(hasNoTypeError("let f(x: f64): f64 = x + 1.0 in f(5.0)"));
}

TEST(TypeCheckTest, LetExpressionWithFunctionInferredReturnType) {
  // Return type should be inferred from function body
  EXPECT_TRUE(hasNoTypeError("let f(x: i64) = x * 2 in f(5)"));
}

TEST(TypeCheckTest, LetExpressionWithFunctionWrongArgType) {
  // Passing wrong type to function in let expression should fail
  EXPECT_TRUE(hasTypeError("let f(x: i64): i64 = x + 1 in f(5.0)"));
}

TEST(TypeCheckTest, LetExpressionWithFunctionReturnTypeMismatch) {
  // Function return type doesn't match body
  EXPECT_TRUE(hasTypeError("let f(x: i64): f64 = x + 1 in f(5)"));
}

TEST(TypeCheckTest, LetExpressionMultipleFunctions) {
  // Multiple functions in let expression
  EXPECT_TRUE(hasNoTypeError(
      "let square(n: i64): i64 = n * n and cube(n: i64): i64 = n * n * n in "
      "square(2) + cube(2)"));
}

TEST(TypeCheckTest, LetExpressionMixedBindingsTypes) {
  // Variable and function in same let expression
  EXPECT_TRUE(hasNoTypeError("let x = 10 and f(y: i64): i64 = y * 2 in f(x)"));
}

// ============== Let Expression Parallel Binding Tests ==============

TEST(TypeCheckTest, LetBindingCannotReferToSibling) {
  // Bindings in let...and cannot refer to each other
  EXPECT_TRUE(hasTypeError("let x = 10 and y = x in y"));
  EXPECT_TRUE(hasTypeError("let a = 1 and b = a + 1 in b"));
}

TEST(TypeCheckTest, LetBindingCanReferToOuterScope) {
  // Nested let can see outer binding
  EXPECT_TRUE(hasNoTypeError("let x = 10 in let y = x in y"));
}

TEST(TypeCheckTest, LetBindingParallelEvaluation) {
  // Both bindings evaluated in same scope - neither sees the other
  EXPECT_TRUE(hasTypeError("let x = y and y = 1 in x"));
}

// ============== Closure / Variable Capture Tests ==============

TEST(TypeCheckTest, SimpleClosure) {
  // Function can capture variable from outer scope
  EXPECT_TRUE(hasNoTypeError("let x = 10\nlet f() = x + 1\nf()"));
}

TEST(TypeCheckTest, ClosureWithParameter) {
  // Function with parameter can also capture
  EXPECT_TRUE(hasNoTypeError(
      "let multiplier = 3\nlet scale(n: i64) = n * multiplier\nscale(5)"));
}

TEST(TypeCheckTest, MultipleCapturedVariables) {
  // Function can capture multiple variables
  EXPECT_TRUE(hasNoTypeError("let a = 1\nlet b = 2\nlet sum() = a + b\nsum()"));
}

TEST(TypeCheckTest, ClosureInLetExpression) {
  // Function in let expression captures sibling variable
  EXPECT_TRUE(hasNoTypeError("let x = 10 and f() = x + 1 in f()"));
}

TEST(TypeCheckTest, ClosureCapturesMutableVariable) {
  // Function can capture mutable variable (requires explicit dereference)
  EXPECT_TRUE(hasNoTypeError("let mut x = 10\nlet f() = (*x) + 1\nf()"));
}

TEST(TypeCheckTest, ClosureTypeMismatch) {
  // Captured variable type must be compatible with usage
  EXPECT_TRUE(hasTypeError("let x = 10\nlet f(): f64 = x + 1.0\nf()"));
}

TEST(TypeCheckTest, ClosureUndeclaredCapture) {
  // Cannot capture undeclared variable
  EXPECT_TRUE(hasTypeError("let f() = y + 1\nf()"));
}

TEST(TypeCheckTest, ClosureCapturesDoubleType) {
  // Function can capture double variable
  EXPECT_TRUE(hasNoTypeError("let x = 3.14\nlet f() = x + 1.0\nf()"));
}

TEST(TypeCheckTest, ClosureCapturesBoolType) {
  // Function can capture bool variable
  EXPECT_TRUE(
      hasNoTypeError("let flag = true\nlet f() = if flag then 1 else 0\nf()"));
}

TEST(TypeCheckTest, ClosureWithParamsAndCaptures) {
  // Function uses both parameters and captured variables
  EXPECT_TRUE(
      hasNoTypeError("let base = 100\nlet add(x: i64) = x + base\nadd(5)"));
}

TEST(TypeCheckTest, NestedLetWithClosure) {
  // Closure in nested let expression
  EXPECT_TRUE(hasNoTypeError("let x = 10 in let f() = x + 1 in f()"));
}

TEST(TypeCheckTest, ClosureInLetWithMultipleSiblings) {
  // Function captures from multiple sibling bindings
  EXPECT_TRUE(hasNoTypeError("let a = 1 and b = 2 and sum() = a + b in sum()"));
}

TEST(TypeCheckTest, ClosureCaptureFromOuterNotSibling) {
  // Function captures from outer scope, not sibling
  EXPECT_TRUE(
      hasNoTypeError("let outer = 5\nlet x = 10 and f() = outer + 1 in f()"));
}

// ============== FreeVariableCollector Tests ==============
// These tests specifically exercise the capture analysis paths

TEST(TypeCheckTest, ClosureWithAssignment) {
  // Assignment inside closure - captures mutable variable
  EXPECT_TRUE(hasNoTypeError("let mut x = 10\nlet f() = x <- 20\nf()"));
}

TEST(TypeCheckTest, ClosureWithAssignmentAndCapture) {
  // Assignment RHS captures another variable
  EXPECT_TRUE(
      hasNoTypeError("let y = 5\nlet mut x = 10\nlet f() = x <- y\nf()"));
}

TEST(TypeCheckTest, ClosureWithLetExpression) {
  // Let expression inside closure that captures outer variable
  EXPECT_TRUE(hasNoTypeError(
      "let outer = 10\nlet f() = let inner = 1 in inner + outer\nf()"));
}

TEST(TypeCheckTest, ClosureWithLetExpressionFunction) {
  // Let expression with function binding inside closure
  EXPECT_TRUE(hasNoTypeError(
      "let outer = 10\nlet f() = let g(x: i64) = x in g(outer)\nf()"));
}

TEST(TypeCheckTest, ClosureWithLetExpressionCaptureInInit) {
  // Capture in let expression initializer inside closure
  EXPECT_TRUE(
      hasNoTypeError("let outer = 5\nlet f() = let x = outer + 1 in x\nf()"));
}

TEST(TypeCheckTest, ClosureWithNestedLetBindings) {
  // Multiple bindings in let inside closure
  EXPECT_TRUE(hasNoTypeError(
      "let outer = 10\nlet f() = let a = 1 and b = outer in a + b\nf()"));
}

TEST(TypeCheckTest, ClosureWithBinaryOpCapture) {
  // Binary operator with captures on both sides
  EXPECT_TRUE(hasNoTypeError("let a = 1\nlet b = 2\nlet f() = a + b\nf()"));
}

TEST(TypeCheckTest, ClosureWithIfConditionCapture) {
  // If condition captures variable
  EXPECT_TRUE(
      hasNoTypeError("let flag = true\nlet f() = if flag then 1 else 0\nf()"));
}

TEST(TypeCheckTest, ClosureWithIfBranchCapture) {
  // If branches capture variables
  EXPECT_TRUE(hasNoTypeError(
      "let x = 1\nlet y = 2\nlet f() = if true then x else y\nf()"));
}

TEST(TypeCheckTest, ClosureWithMethodCallArgs) {
  // Method call arguments capture variables
  EXPECT_TRUE(hasNoTypeError(
      "let x = 5\nlet add(a: i64, b: i64) = a + b\nlet f() = add(x, x)\nf()"));
}

TEST(TypeCheckTest, ClosureWithNestedBlocks) {
  // Block with multiple expression statements
  EXPECT_TRUE(hasNoTypeError("let x = 1\nlet f() = x + 1\nf()"));
}

TEST(TypeCheckTest, ClosureDoesNotCaptureLocalLetBinding) {
  // Local let binding should not be captured
  EXPECT_TRUE(hasNoTypeError("let f() = let local = 5 in local + 1\nf()"));
}

TEST(TypeCheckTest, ClosureWithMutableAssignmentCapture) {
  // Capture mutable variable via assignment LHS (requires explicit dereference)
  EXPECT_TRUE(hasNoTypeError(
      "let mut counter = 0\nlet inc() = counter <- (*counter) + 1\ninc()"));
}
