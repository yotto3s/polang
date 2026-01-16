#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "integration_test_helper.hpp"

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
