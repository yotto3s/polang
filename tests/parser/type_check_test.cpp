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
  EXPECT_TRUE(hasNoTypeError("x = 1 + 2"));
  EXPECT_TRUE(hasNoTypeError("x = 1 - 2"));
  EXPECT_TRUE(hasNoTypeError("x = 1 * 2"));
  EXPECT_TRUE(hasNoTypeError("x = 1 / 2"));
}

TEST(TypeCheckTest, DoubleArithmetic) {
  EXPECT_TRUE(hasNoTypeError("x : f64\nx = 1.0 + 2.0"));
  EXPECT_TRUE(hasNoTypeError("x : f64\nx = 1.0 - 2.0"));
  EXPECT_TRUE(hasNoTypeError("x : f64\nx = 1.0 * 2.0"));
  EXPECT_TRUE(hasNoTypeError("x : f64\nx = 1.0 / 2.0"));
}

TEST(TypeCheckTest, VariableUsage) {
  EXPECT_TRUE(hasNoTypeError("x = 1\ny = x + 1"));
  EXPECT_TRUE(hasNoTypeError("x : f64\nx = 1.0\ny : f64\ny = x + 2.0"));
}

TEST(TypeCheckTest, FunctionDeclaration) {
  EXPECT_TRUE(
      hasNoTypeError("add : i64 * i64 -> i64\nadd(x, y) = x + y"));
  EXPECT_TRUE(
      hasNoTypeError("mul : f64 * f64 -> f64\nmul(a, b) = a * b"));
}

TEST(TypeCheckTest, LetExpression) {
  EXPECT_TRUE(hasNoTypeError("let x = 1 in x + 1"));
  EXPECT_TRUE(hasNoTypeError("let x = 1 and y = 2 in x + y"));
}

TEST(TypeCheckTest, IfExpression) {
  // If condition must be bool (comparison or boolean literal)
  EXPECT_TRUE(hasNoTypeError("if true then 2 else 3"));
  EXPECT_TRUE(hasNoTypeError("if 1 == 1 then 2 else 3"));
  EXPECT_TRUE(hasNoTypeError("x = if true then 2 else 3"));
}

TEST(TypeCheckTest, Comparison) {
  // Comparisons return bool, so variable must be declared as bool
  EXPECT_TRUE(hasNoTypeError("x : bool\nx = 1 < 2"));
  EXPECT_TRUE(hasNoTypeError("x : bool\nx = 1 == 2"));
  EXPECT_TRUE(hasNoTypeError(
      "x : f64\nx = 1.0\ny : f64\ny = 2.0\nz : bool\nz = x < y"));
  // Using comparison in if condition (returns bool)
  EXPECT_TRUE(hasNoTypeError("if 1 < 2 then 3 else 4"));
}

// ============== Type Error Tests ==============

TEST(TypeCheckTest, MixedArithmeticTypes) {
  EXPECT_TRUE(hasTypeError("x = 1 + 2.0"));
  EXPECT_TRUE(hasTypeError("x = 1.0 - 2"));
  EXPECT_TRUE(hasTypeError("x = 1 * 2.0"));
  EXPECT_TRUE(hasTypeError("x = 1.0 / 2"));
}

TEST(TypeCheckTest, MixedComparisonTypes) {
  EXPECT_TRUE(hasTypeError("x = 1 < 2.0"));
  EXPECT_TRUE(hasTypeError("x = 1.0 == 2"));
}

TEST(TypeCheckTest, VariableTypeMismatch) {
  EXPECT_TRUE(hasTypeError("x : i64\nx = 1.0"));
  EXPECT_TRUE(hasTypeError("x : f64\nx = 1"));
}

TEST(TypeCheckTest, FunctionReturnTypeMismatch) {
  EXPECT_TRUE(hasTypeError("f : i64 -> f64\nf(x) = x"));
  EXPECT_TRUE(hasTypeError("f : f64 -> i64\nf(x) = x"));
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
  EXPECT_TRUE(hasTypeError("y = x"));
}

TEST(TypeCheckTest, LetExpressionTypeMismatch) {
  EXPECT_TRUE(hasTypeError("let x = 1 in x + 2.0"));
  EXPECT_TRUE(hasTypeError("let x : i64 = 1 and y : f64 = 2.0 in x + y"));
}

// ============== Error Message Tests ==============

TEST(TypeCheckTest, ErrorMessageContainsOperator) {
  auto errors = checkTypes("x = 1 + 2.0");
  ASSERT_FALSE(errors.empty());
  EXPECT_TRUE(errors[0].message.find("+") != std::string::npos);
}

TEST(TypeCheckTest, ErrorMessageContainsTypes) {
  auto errors = checkTypes("x = 1 + 2.0");
  ASSERT_FALSE(errors.empty());
  EXPECT_TRUE(errors[0].message.find("i64") != std::string::npos);
  EXPECT_TRUE(errors[0].message.find("f64") != std::string::npos);
}

// ============== Type Inference Tests ==============

TEST(TypeCheckTest, InferIntFromLiteral) {
  // x = 42 should infer int
  EXPECT_TRUE(hasNoTypeError("x = 42\ny : i64\ny = x"));
}

TEST(TypeCheckTest, InferDoubleFromLiteral) {
  // x = 3.14 should infer double
  EXPECT_TRUE(hasNoTypeError("x = 3.14\ny : f64\ny = x"));
}

TEST(TypeCheckTest, InferBoolFromLiteral) {
  // x = true should infer bool
  EXPECT_TRUE(hasNoTypeError("x = true\nif x then 1 else 0"));
}

TEST(TypeCheckTest, InferFromExpression) {
  // x = 1 + 2 should infer int
  EXPECT_TRUE(hasNoTypeError("x = 1 + 2\ny : i64\ny = x"));
}

TEST(TypeCheckTest, InferFromComparison) {
  // x = 1 < 2 should infer bool
  EXPECT_TRUE(hasNoTypeError("x = 1 < 2\nif x then 1 else 0"));
}

TEST(TypeCheckTest, InferFunctionReturnType) {
  // f(x) = x + 1 should infer int return type
  EXPECT_TRUE(hasNoTypeError(
      "f : i64 -> i64\nf(x) = x + 1\ny : i64\ny = f(5)"));
}

TEST(TypeCheckTest, InferFunctionReturnTypeDouble) {
  // f(x) = x + 1.0 should infer double return type
  EXPECT_TRUE(hasNoTypeError(
      "f : f64 -> f64\nf(x) = x + 1.0\ny : f64\ny = f(5.0)"));
}

TEST(TypeCheckTest, NoImplicitConversionIntToDouble) {
  // x: f64 = 42 should be error (no coercion)
  EXPECT_TRUE(hasTypeError("x : f64\nx = 42"));
}

TEST(TypeCheckTest, NoImplicitConversionDoubleToInt) {
  // x: i64 = 42.0 should be error (no coercion)
  EXPECT_TRUE(hasTypeError("x : i64\nx = 42.0"));
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
  // x = 42 followed by double operation should fail
  EXPECT_TRUE(hasTypeError("x = 42\ny = x + 1.0"));
}

// ============== Function Call Type Checking Tests ==============

TEST(TypeCheckTest, FunctionCallCorrectTypes) {
  // Correct argument types should pass
  EXPECT_TRUE(
      hasNoTypeError("f : i64 -> i64\nf(x) = x + 1\nf(5)"));
  EXPECT_TRUE(
      hasNoTypeError("f : f64 -> f64\nf(x) = x + 1.0\nf(5.0)"));
  EXPECT_TRUE(hasNoTypeError(
      "f : i64 * i64 -> i64\nf(x, y) = x + y\nf(1, 2)"));
}

TEST(TypeCheckTest, FunctionCallWrongArgType) {
  // Passing double to int parameter should fail
  EXPECT_TRUE(
      hasTypeError("f : i64 -> i64\nf(x) = x + 1\nf(3.5)"));
  // Passing int to double parameter should fail
  EXPECT_TRUE(
      hasTypeError("f : f64 -> f64\nf(x) = x + 1.0\nf(3)"));
}

TEST(TypeCheckTest, FunctionCallWrongArgCount) {
  // Too few arguments
  EXPECT_TRUE(hasTypeError(
      "f : i64 * i64 -> i64\nf(x, y) = x + y\nf(1)"));
  // Too many arguments
  EXPECT_TRUE(
      hasTypeError("f : i64 -> i64\nf(x) = x + 1\nf(1, 2)"));
}

TEST(TypeCheckTest, FunctionCallMultipleArgsTypeMismatch) {
  // Second argument has wrong type
  EXPECT_TRUE(hasTypeError(
      "f : i64 * i64 -> i64\nf(x, y) = x + y\nf(1, 2.0)"));
  // First argument has wrong type
  EXPECT_TRUE(hasTypeError(
      "f : i64 * i64 -> i64\nf(x, y) = x + y\nf(1.0, 2)"));
}

TEST(TypeCheckTest, FunctionCallErrorMessage) {
  auto errors = checkTypes("f : i64 -> i64\nf(x) = x + 1\nf(3.5)");
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
  EXPECT_TRUE(hasNoTypeError("x = 10\nf() = x + 1\nf()"));
}

TEST(TypeCheckTest, ClosureWithParameter) {
  // Function with parameter can also capture
  EXPECT_TRUE(hasNoTypeError(
      "multiplier = 3\nscale : i64 -> i64\nscale(n) = n * multiplier\nscale(5)"));
}

TEST(TypeCheckTest, MultipleCapturedVariables) {
  // Function can capture multiple variables
  EXPECT_TRUE(hasNoTypeError("a = 1\nb = 2\nsum() = a + b\nsum()"));
}

TEST(TypeCheckTest, ClosureInLetExpression) {
  // Function in let expression captures sibling variable
  EXPECT_TRUE(hasNoTypeError("let x = 10 and f() = x + 1 in f()"));
}

TEST(TypeCheckTest, ClosureTypeMismatch) {
  // Captured variable type must be compatible with usage
  EXPECT_TRUE(hasTypeError("x = 10\nf() = x + 1.0\nf()"));
}

TEST(TypeCheckTest, ClosureUndeclaredCapture) {
  // Cannot capture undeclared variable
  EXPECT_TRUE(hasTypeError("f() = y + 1\nf()"));
}

TEST(TypeCheckTest, ClosureCapturesDoubleType) {
  // Function can capture double variable
  EXPECT_TRUE(hasNoTypeError("x = 3.14\nf() = x + 1.0\nf()"));
}

TEST(TypeCheckTest, ClosureCapturesBoolType) {
  // Function can capture bool variable
  EXPECT_TRUE(
      hasNoTypeError("flag = true\nf() = if flag then 1 else 0\nf()"));
}

TEST(TypeCheckTest, ClosureWithParamsAndCaptures) {
  // Function uses both parameters and captured variables
  EXPECT_TRUE(hasNoTypeError(
      "base = 100\nadd : i64 -> i64\nadd(x) = x + base\nadd(5)"));
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
      hasNoTypeError("outer = 5\nlet x = 10 and f() = outer + 1 in f()"));
}

// ============== FreeVariableCollector Tests ==============
// These tests specifically exercise the capture analysis paths

TEST(TypeCheckTest, ClosureWithLetExpression) {
  // Let expression inside closure that captures outer variable
  EXPECT_TRUE(hasNoTypeError(
      "outer = 10\nf() = let inner = 1 in inner + outer\nf()"));
}

TEST(TypeCheckTest, ClosureWithLetExpressionFunction) {
  // Let expression with function binding inside closure
  EXPECT_TRUE(hasNoTypeError(
      "outer = 10\nf() = let g(x: i64) = x in g(outer)\nf()"));
}

TEST(TypeCheckTest, ClosureWithLetExpressionCaptureInInit) {
  // Capture in let expression initializer inside closure
  EXPECT_TRUE(
      hasNoTypeError("outer = 5\nf() = let x = outer + 1 in x\nf()"));
}

TEST(TypeCheckTest, ClosureWithNestedLetBindings) {
  // Multiple bindings in let inside closure
  EXPECT_TRUE(hasNoTypeError(
      "outer = 10\nf() = let a = 1 and b = outer in a + b\nf()"));
}

TEST(TypeCheckTest, ClosureWithBinaryOpCapture) {
  // Binary operator with captures on both sides
  EXPECT_TRUE(hasNoTypeError("a = 1\nb = 2\nf() = a + b\nf()"));
}

TEST(TypeCheckTest, ClosureWithIfConditionCapture) {
  // If condition captures variable
  EXPECT_TRUE(
      hasNoTypeError("flag = true\nf() = if flag then 1 else 0\nf()"));
}

TEST(TypeCheckTest, ClosureWithIfBranchCapture) {
  // If branches capture variables
  EXPECT_TRUE(hasNoTypeError(
      "x = 1\ny = 2\nf() = if true then x else y\nf()"));
}

TEST(TypeCheckTest, ClosureWithMethodCallArgs) {
  // Method call arguments capture variables
  EXPECT_TRUE(hasNoTypeError(
      "x = 5\nadd : i64 * i64 -> i64\nadd(a, b) = a + b\nf() = add(x, x)\nf()"));
}

TEST(TypeCheckTest, ClosureWithNestedBlocks) {
  // Block with multiple expression statements
  EXPECT_TRUE(hasNoTypeError("x = 1\nf() = x + 1\nf()"));
}

TEST(TypeCheckTest, ClosureDoesNotCaptureLocalLetBinding) {
  // Local let binding should not be captured
  EXPECT_TRUE(hasNoTypeError("f() = let local = 5 in local + 1\nf()"));
}

// ============== Unit Type Signature Tests ==============

TEST(TypeCheckTest, UnitTypeSignatureValid) {
  EXPECT_TRUE(hasNoTypeError("f : () -> i64\nf() = 42"));
  EXPECT_TRUE(hasNoTypeError("g : () -> bool\ng() = true"));
  EXPECT_TRUE(hasNoTypeError("h : () -> f64\nh() = 3.14"));
}

TEST(TypeCheckTest, BareReturnTypeRejected) {
  // f : i64 with f() = ... should now be an error
  EXPECT_TRUE(hasTypeError("f : i64\nf() = 42"));
}

TEST(TypeCheckTest, UnitTypeArityMismatch) {
  // () -> T with f(x) = ... should be an error
  EXPECT_TRUE(hasTypeError("f : () -> i64\nf(x) = x"));
}
