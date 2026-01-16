#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "integration_test_helper.hpp"

// Test exit code verification - verify exit code 0 for valid programs

TEST(ReplIntegration, IntegerLiteral) {
  const auto result = runRepl("42");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ReplIntegration, DoubleLiteral) {
  const auto result = runRepl("3.14");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ReplIntegration, BooleanLiteral) {
  const auto result = runRepl("true");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ReplIntegration, BinaryOperation) {
  const auto result = runRepl("1 + 2");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ReplIntegration, VariableDeclaration) {
  const auto result = runRepl("let x: int = 5");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ReplIntegration, FunctionDeclarationAndCall) {
  const auto result = runRepl("let double (x: int): int = x * 2\ndouble(5)");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ReplIntegration, IfExpression) {
  const auto result = runRepl("if true then 1 else 2");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ReplIntegration, LetExpression) {
  const auto result = runRepl("let x = 1 in x + 1");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ReplIntegration, NestedLetExpression) {
  const auto result = runRepl("let x = 1 in let y = 2 in x + y");
  EXPECT_EQ(result.exit_code, 0);
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
