#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "process_helper.hpp"

using ::testing::HasSubstr;

// Basic literal, operation, and error tests are covered by lit tests in tests/lit/
// See: tests/lit/Execution/, tests/lit/Errors/

// ============== Function in Let Expression Tests ==============

TEST(ReplIntegration, LetExpressionWithFunction) {
  const auto result = runRepl("let f(x: int): int = x + 1 in f(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("6"));
  EXPECT_THAT(result.stdout_output, HasSubstr("int"));
}

TEST(ReplIntegration, LetExpressionMultipleFunctions) {
  const auto result = runRepl(
      "let square(n: int): int = n * n and cube(n: int): int = n * n * n in "
      "square(3) + cube(2)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("17")); // 9 + 8
  EXPECT_THAT(result.stdout_output, HasSubstr("int"));
}

TEST(ReplIntegration, LetExpressionMixedBindingsCallBody) {
  // Variable and function in let, function is called with variable
  const auto result = runRepl("let x = 10 and f(y: int): int = y * 2 in f(x)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("20"));
  EXPECT_THAT(result.stdout_output, HasSubstr("int"));
}

// ============== State Persistence Tests ==============
// These tests verify that variables and functions persist across evaluations

TEST(ReplIntegration, VariablePersistsAcrossEvaluations) {
  // Declare variable, then use it in next evaluation
  const auto result = runRepl("let x = 42\nx + 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("43 : int"));
}

TEST(ReplIntegration, FunctionPersistsAcrossEvaluations) {
  // Declare function, then call it in next evaluation
  const auto result = runRepl("let double(n: int): int = n * 2\ndouble(21)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("42 : int"));
}

TEST(ReplIntegration, MultipleVariablesPersist) {
  // Multiple variables across multiple lines
  const auto result = runRepl("let a = 10\nlet b = 20\nlet c = 30\na + b + c");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("60 : int"));
}

TEST(ReplIntegration, FunctionCalledWithPersistedVariable) {
  // Variable declared, then passed to function call
  const auto result = runRepl("let multiplier = 3\nlet scale(x: int, y: int): "
                              "int = x * y\nscale(10, multiplier)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("30 : int"));
}

TEST(ReplIntegration, MultipleFunctionsPersist) {
  // Multiple functions, each can call the other
  const auto result = runRepl("let add1(x: int): int = x + 1\n"
                              "let add2(x: int): int = add1(add1(x))\n"
                              "add2(10)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("12 : int"));
}

TEST(ReplIntegration, VariableReassignmentPersists) {
  // Reassign variable and verify new value persists
  const auto result = runRepl("let mut x = 1\nx <- 100\nx");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("100 : int"));
}

// Assignment return value tests (AssignmentReturnsIntValue, AssignmentReturnsDoubleValue,
// AssignmentReturnsBoolValue, ChainedAssignment) are covered by lit tests in tests/lit/Execution/

// Multi-line input detection tests are covered by repl_unit_test.cpp (IsInputIncomplete tests)

// ============== Closure / Variable Capture Tests ==============

TEST(ReplIntegration, SimpleClosure) {
  const auto result = runRepl("let x = 10\nlet f() = x + 1\nf()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("11 : int"));
}

TEST(ReplIntegration, ClosureWithParameter) {
  const auto result = runRepl("let multiplier = 3\nlet scale(n: int) = n * multiplier\nscale(10)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("30 : int"));
}

TEST(ReplIntegration, ClosureInLetExpression) {
  const auto result = runRepl("let x = 10 and f() = x + 1 in f()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("11 : int"));
}

TEST(ReplIntegration, MultipleCapturedVariables) {
  const auto result = runRepl("let a = 1\nlet b = 2\nlet sum() = a + b\nsum()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3 : int"));
}

TEST(ReplIntegration, ClosureWithMutableCapture) {
  const auto result = runRepl("let mut x = 10\nlet f() = x + 1\nf()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("11 : int"));
}

TEST(ReplIntegration, ClosureCaptureByValue) {
  // Capture is by value at call time, so mutation after function definition
  // should be visible when calling
  const auto result = runRepl("let mut x = 10\nlet f() = x + 1\nx <- 20\nf()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("21 : int"));
}

TEST(ReplIntegration, ClosureCapturesDouble) {
  const auto result = runRepl("let x = 3.14\nlet f() = x + 1.0\nf()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("4.14 : double"));
}

TEST(ReplIntegration, ClosureWithParamsAndCaptures) {
  const auto result = runRepl("let base = 100\nlet add(x: int) = x + base\nadd(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("105 : int"));
}

TEST(ReplIntegration, NestedLetWithClosure) {
  const auto result = runRepl("let x = 10 in let f() = x + 1 in f()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("11 : int"));
}

TEST(ReplIntegration, ClosureMultipleSiblings) {
  const auto result = runRepl("let a = 1 and b = 2 and sum() = a + b in sum()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3 : int"));
}
