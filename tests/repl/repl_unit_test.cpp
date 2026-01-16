#include <gtest/gtest.h>

#include "repl/input_checker.hpp"

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
