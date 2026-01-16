// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
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
  NBlock* block = polang_parse(source);
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
  EXPECT_TRUE(hasNoTypeError("let x : double = 1.0 + 2.0"));
  EXPECT_TRUE(hasNoTypeError("let x : double = 1.0 - 2.0"));
  EXPECT_TRUE(hasNoTypeError("let x : double = 1.0 * 2.0"));
  EXPECT_TRUE(hasNoTypeError("let x : double = 1.0 / 2.0"));
}

TEST(TypeCheckTest, VariableUsage) {
  EXPECT_TRUE(hasNoTypeError("let x = 1\nlet y = x + 1"));
  EXPECT_TRUE(hasNoTypeError("let x : double = 1.0\nlet y : double = x + 2.0"));
}

TEST(TypeCheckTest, FunctionDeclaration) {
  EXPECT_TRUE(hasNoTypeError("let add (x : int) (y : int) : int = x + y"));
  EXPECT_TRUE(hasNoTypeError("let mul (a : double) (b : double) : double = a * b"));
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
  EXPECT_TRUE(hasNoTypeError("let x : double = 1.0\nlet y : double = 2.0\nlet z : bool = x < y"));
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
  EXPECT_TRUE(hasTypeError("let x : int = 1.0"));
  EXPECT_TRUE(hasTypeError("let x : double = 1"));
}

TEST(TypeCheckTest, FunctionReturnTypeMismatch) {
  EXPECT_TRUE(hasTypeError("let f (x : int) : double = x"));
  EXPECT_TRUE(hasTypeError("let f (x : double) : int = x"));
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
  EXPECT_TRUE(hasTypeError("let x : int = 1 and y : double = 2.0 in x + y"));
}

TEST(TypeCheckTest, AssignmentTypeMismatch) {
  EXPECT_TRUE(hasTypeError("let x = 1\nx = 2.0"));
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
  EXPECT_TRUE(errors[0].message.find("int") != std::string::npos);
  EXPECT_TRUE(errors[0].message.find("double") != std::string::npos);
}
