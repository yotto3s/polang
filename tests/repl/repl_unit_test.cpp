#include <gtest/gtest.h>

#include "repl/input_checker.hpp"
#include "repl/repl_session.hpp"

// ============== isInputIncomplete Unit Tests ==============
// Direct tests for the InputChecker::isInputIncomplete function

// Parentheses tests
TEST(IsInputIncomplete, UnbalancedOpenParen) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("(1 + 2"));
}

TEST(IsInputIncomplete, BalancedParens) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("(1 + 2)"));
}

TEST(IsInputIncomplete, NestedUnbalancedParens) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("((1 + 2)"));
}

TEST(IsInputIncomplete, NestedBalancedParens) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("((1 + 2))"));
}

TEST(IsInputIncomplete, MultipleParenGroups) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("(1) + (2)"));
}

// If/else tests
TEST(IsInputIncomplete, IfWithoutElse) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("if true then 1"));
}

TEST(IsInputIncomplete, CompleteIfElse) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("if true then 1 else 2"));
}

TEST(IsInputIncomplete, NestedIfWithoutOuterElse) {
  EXPECT_TRUE(
      InputChecker::isInputIncomplete("if true then if false then 1 else 2"));
}

TEST(IsInputIncomplete, NestedIfComplete) {
  EXPECT_FALSE(InputChecker::isInputIncomplete(
      "if true then if false then 1 else 2 else 3"));
}

// Trailing keyword tests
TEST(IsInputIncomplete, TrailingIn) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("let x = 5 in"));
}

TEST(IsInputIncomplete, TrailingThen) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("if true then"));
}

TEST(IsInputIncomplete, TrailingAnd) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("let x = 1 and"));
}

TEST(IsInputIncomplete, TrailingComma) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("f(1,"));
}

// Trailing operator tests
TEST(IsInputIncomplete, TrailingPlus) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("1 +"));
}

TEST(IsInputIncomplete, TrailingMinus) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("1 -"));
}

TEST(IsInputIncomplete, TrailingMultiply) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("1 *"));
}

TEST(IsInputIncomplete, TrailingDivide) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("1 /"));
}

TEST(IsInputIncomplete, TrailingEquals) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("x ="));
}

// Complete expression tests
TEST(IsInputIncomplete, SimpleLiteral) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("42"));
}

TEST(IsInputIncomplete, SimpleExpression) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("1 + 2"));
}

TEST(IsInputIncomplete, VariableDeclaration) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("let x = 5"));
}

TEST(IsInputIncomplete, FunctionDeclaration) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("let f(x: int): int = x + 1"));
}

TEST(IsInputIncomplete, LetExpression) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("let x = 1 in x + 1"));
}

TEST(IsInputIncomplete, FunctionCall) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("f(1, 2)"));
}

// Whitespace handling
TEST(IsInputIncomplete, WhitespaceAfterOperator) {
  EXPECT_TRUE(InputChecker::isInputIncomplete("1 +   "));
}

TEST(IsInputIncomplete, NewlineInMiddle) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("1 + 2\n"));
}

TEST(IsInputIncomplete, EmptyInput) {
  EXPECT_FALSE(InputChecker::isInputIncomplete(""));
}

TEST(IsInputIncomplete, WhitespaceOnly) {
  EXPECT_FALSE(InputChecker::isInputIncomplete("   "));
}

// ============== EvalResult Unit Tests ==============

TEST(EvalResult, OkFactory) {
  const auto result = EvalResult::ok();
  EXPECT_TRUE(result.success);
  EXPECT_FALSE(result.hasValue);
  EXPECT_EQ(result.type, "void");
  EXPECT_TRUE(result.errorMessage.empty());
}

TEST(EvalResult, ValueFactory) {
  const auto result = EvalResult::value(42, "int");
  EXPECT_TRUE(result.success);
  EXPECT_TRUE(result.hasValue);
  EXPECT_EQ(result.rawValue, 42);
  EXPECT_EQ(result.type, "int");
  EXPECT_TRUE(result.errorMessage.empty());
}

TEST(EvalResult, ErrorFactory) {
  const auto result = EvalResult::error("test error message");
  EXPECT_FALSE(result.success);
  EXPECT_FALSE(result.hasValue);
  EXPECT_EQ(result.errorMessage, "test error message");
}
