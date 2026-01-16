#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "process_helper.hpp"

using ::testing::HasSubstr;

// Test output verification - verify result value and type

TEST(ReplIntegration, IntegerLiteral) {
  const auto result = runRepl("42");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("42 : int"));
}

TEST(ReplIntegration, DoubleLiteral) {
  const auto result = runRepl("3.14");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3.14 : double"));
}

TEST(ReplIntegration, BooleanLiteral) {
  const auto result = runRepl("true");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("true : bool"));
}

TEST(ReplIntegration, BooleanLiteralFalse) {
  const auto result = runRepl("false");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("false : bool"));
}

TEST(ReplIntegration, BinaryOperation) {
  const auto result = runRepl("1 + 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3 : int"));
}

TEST(ReplIntegration, DoubleBinaryOperation) {
  const auto result = runRepl("1.5 + 2.5");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("4 : double"));
}

TEST(ReplIntegration, ComparisonOperation) {
  const auto result = runRepl("1 == 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("true : bool"));
}

TEST(ReplIntegration, DoubleComparisonOperation) {
  const auto result = runRepl("1.5 < 2.5");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("true : bool"));
}

TEST(ReplIntegration, VariableDeclaration) {
  const auto result = runRepl("let x: int = 5");
  EXPECT_EQ(result.exit_code, 0);
  // Variable declaration returns alloca which is not printed as known type
}

TEST(ReplIntegration, FunctionDeclarationAndCall) {
  const auto result = runRepl("let double(x: int): int = x * 2\ndouble(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("10 : int"));
}

TEST(ReplIntegration, IfExpression) {
  const auto result = runRepl("if true then 1 else 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("1 : int"));
}

TEST(ReplIntegration, LetExpression) {
  const auto result = runRepl("let x = 1 in x + 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("2 : int"));
}

TEST(ReplIntegration, NestedLetExpression) {
  const auto result = runRepl("let x = 1 in let y = 2 in x + y");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3 : int"));
}

// Test error handling - verify exit code 1 on errors

TEST(ReplIntegration, SyntaxError) {
  const auto result = runRepl("let x =");
  EXPECT_EQ(result.exit_code, 1);
}

TEST(ReplIntegration, TypeError) {
  const auto result = runRepl("1 + 1.0");
  EXPECT_EQ(result.exit_code, 1);
}

TEST(ReplIntegration, TypeErrorIfCondition) {
  const auto result = runRepl("if 1 then 2 else 3");
  EXPECT_EQ(result.exit_code, 1);
}

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

TEST(ReplIntegration, AssignmentReturnsIntValue) {
  // Assignment expression returns the assigned value
  const auto result = runRepl("let mut x = 0\nx <- 42");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("42 : int"));
}

TEST(ReplIntegration, AssignmentReturnsDoubleValue) {
  // Assignment expression returns double value
  const auto result = runRepl("let mut x: double = 0.0\nx <- 3.14");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3.14 : double"));
}

TEST(ReplIntegration, AssignmentReturnsBoolValue) {
  // Assignment expression returns bool value
  const auto result = runRepl("let mut flag: bool = false\nflag <- true");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("true : bool"));
}

TEST(ReplIntegration, ChainedAssignment) {
  // Chained assignment: x <- y <- 5 assigns 5 to both
  const auto result = runRepl("let mut x = 0\nlet mut y = 0\nx <- y <- 5\nx + y");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("10 : int"));
}

// ============== Multi-line Input Tests ==============
// These tests verify the isInputIncomplete detection works correctly

TEST(ReplIntegration, MultiLineIfExpression) {
  // If expression split across lines (if without else is incomplete)
  const auto result = runRepl("if true then 1\nelse 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("1 : int"));
}

TEST(ReplIntegration, MultiLineLetExpression) {
  // Let expression split across lines (let without in body)
  const auto result = runRepl("let x = 5 in\nx + 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("6 : int"));
}

TEST(ReplIntegration, MultiLineWithParentheses) {
  // Expression with unbalanced parentheses continues
  const auto result = runRepl("(1 +\n2) * 3");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("9 : int"));
}

TEST(ReplIntegration, MultiLineFunction) {
  // Function declaration and call split across lines
  const auto result = runRepl("let f(x: int): int =\nx * 2\nf(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("10 : int"));
}

TEST(ReplIntegration, MultiLineWithAndKeyword) {
  // Let with 'and' keyword continues to next line
  const auto result = runRepl("let x = 1 and\ny = 2 in x + y");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3 : int"));
}

TEST(ReplIntegration, ComplexMultiLineExpression) {
  // Complex expression across multiple lines
  const auto result = runRepl("let sum(a: int, b: int): int = a + b\n"
                              "let x = 10\n"
                              "sum(x, 20)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("30 : int"));
}

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
