#include <gtest/gtest.h>

// clang-format off
#include "parser/operator_utils.hpp"
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison types
// clang-format on

using namespace polang;

// Token type shorthand
using token = yy::parser::token;

// ============== operatorToString Tests ==============

TEST(OperatorUtilsTest, OperatorToStringPlus) {
  EXPECT_EQ(operatorToString(token::TPLUS), "+");
}

TEST(OperatorUtilsTest, OperatorToStringMinus) {
  EXPECT_EQ(operatorToString(token::TMINUS), "-");
}

TEST(OperatorUtilsTest, OperatorToStringMul) {
  EXPECT_EQ(operatorToString(token::TMUL), "*");
}

TEST(OperatorUtilsTest, OperatorToStringDiv) {
  EXPECT_EQ(operatorToString(token::TDIV), "/");
}

TEST(OperatorUtilsTest, OperatorToStringEqual) {
  EXPECT_EQ(operatorToString(token::TCEQ), "==");
}

TEST(OperatorUtilsTest, OperatorToStringNotEqual) {
  EXPECT_EQ(operatorToString(token::TCNE), "!=");
}

TEST(OperatorUtilsTest, OperatorToStringLessThan) {
  EXPECT_EQ(operatorToString(token::TCLT), "<");
}

TEST(OperatorUtilsTest, OperatorToStringLessEqual) {
  EXPECT_EQ(operatorToString(token::TCLE), "<=");
}

TEST(OperatorUtilsTest, OperatorToStringGreaterThan) {
  EXPECT_EQ(operatorToString(token::TCGT), ">");
}

TEST(OperatorUtilsTest, OperatorToStringGreaterEqual) {
  EXPECT_EQ(operatorToString(token::TCGE), ">=");
}

TEST(OperatorUtilsTest, OperatorToStringUnknown) {
  // Unknown operator should return "?"
  EXPECT_EQ(operatorToString(-1), "?");
  EXPECT_EQ(operatorToString(9999), "?");
}

// ============== isArithmeticOperator Tests ==============

TEST(OperatorUtilsTest, IsArithmeticOperatorPlus) {
  EXPECT_TRUE(isArithmeticOperator(token::TPLUS));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorMinus) {
  EXPECT_TRUE(isArithmeticOperator(token::TMINUS));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorMul) {
  EXPECT_TRUE(isArithmeticOperator(token::TMUL));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorDiv) {
  EXPECT_TRUE(isArithmeticOperator(token::TDIV));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorComparison) {
  // Comparison operators are not arithmetic
  EXPECT_FALSE(isArithmeticOperator(token::TCEQ));
  EXPECT_FALSE(isArithmeticOperator(token::TCNE));
  EXPECT_FALSE(isArithmeticOperator(token::TCLT));
  EXPECT_FALSE(isArithmeticOperator(token::TCLE));
  EXPECT_FALSE(isArithmeticOperator(token::TCGT));
  EXPECT_FALSE(isArithmeticOperator(token::TCGE));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorUnknown) {
  EXPECT_FALSE(isArithmeticOperator(-1));
  EXPECT_FALSE(isArithmeticOperator(0));
}

// ============== isComparisonOperator Tests ==============

TEST(OperatorUtilsTest, IsComparisonOperatorEqual) {
  EXPECT_TRUE(isComparisonOperator(token::TCEQ));
}

TEST(OperatorUtilsTest, IsComparisonOperatorNotEqual) {
  EXPECT_TRUE(isComparisonOperator(token::TCNE));
}

TEST(OperatorUtilsTest, IsComparisonOperatorLessThan) {
  EXPECT_TRUE(isComparisonOperator(token::TCLT));
}

TEST(OperatorUtilsTest, IsComparisonOperatorLessEqual) {
  EXPECT_TRUE(isComparisonOperator(token::TCLE));
}

TEST(OperatorUtilsTest, IsComparisonOperatorGreaterThan) {
  EXPECT_TRUE(isComparisonOperator(token::TCGT));
}

TEST(OperatorUtilsTest, IsComparisonOperatorGreaterEqual) {
  EXPECT_TRUE(isComparisonOperator(token::TCGE));
}

TEST(OperatorUtilsTest, IsComparisonOperatorArithmetic) {
  // Arithmetic operators are not comparison
  EXPECT_FALSE(isComparisonOperator(token::TPLUS));
  EXPECT_FALSE(isComparisonOperator(token::TMINUS));
  EXPECT_FALSE(isComparisonOperator(token::TMUL));
  EXPECT_FALSE(isComparisonOperator(token::TDIV));
}

TEST(OperatorUtilsTest, IsComparisonOperatorUnknown) {
  EXPECT_FALSE(isComparisonOperator(-1));
  EXPECT_FALSE(isComparisonOperator(0));
}

// ============== getOperatorCategory Tests ==============

TEST(OperatorUtilsTest, GetOperatorCategoryArithmetic) {
  EXPECT_EQ(getOperatorCategory(token::TPLUS), OperatorCategory::Arithmetic);
  EXPECT_EQ(getOperatorCategory(token::TMINUS), OperatorCategory::Arithmetic);
  EXPECT_EQ(getOperatorCategory(token::TMUL), OperatorCategory::Arithmetic);
  EXPECT_EQ(getOperatorCategory(token::TDIV), OperatorCategory::Arithmetic);
}

TEST(OperatorUtilsTest, GetOperatorCategoryComparison) {
  EXPECT_EQ(getOperatorCategory(token::TCEQ), OperatorCategory::Comparison);
  EXPECT_EQ(getOperatorCategory(token::TCNE), OperatorCategory::Comparison);
  EXPECT_EQ(getOperatorCategory(token::TCLT), OperatorCategory::Comparison);
  EXPECT_EQ(getOperatorCategory(token::TCLE), OperatorCategory::Comparison);
  EXPECT_EQ(getOperatorCategory(token::TCGT), OperatorCategory::Comparison);
  EXPECT_EQ(getOperatorCategory(token::TCGE), OperatorCategory::Comparison);
}

TEST(OperatorUtilsTest, GetOperatorCategoryUnknown) {
  EXPECT_EQ(getOperatorCategory(0), OperatorCategory::Unknown);
  EXPECT_EQ(getOperatorCategory(999), OperatorCategory::Unknown);
}

// ============== Mutual Exclusion Tests ==============

TEST(OperatorUtilsTest, ArithmeticAndComparisonMutuallyExclusive) {
  // No operator should be both arithmetic and comparison
  const int operators[] = {token::TPLUS,  token::TMINUS, token::TMUL,
                           token::TDIV,   token::TCEQ,   token::TCNE,
                           token::TCLT,   token::TCLE,   token::TCGT,
                           token::TCGE};

  for (int op : operators) {
    bool isArith = isArithmeticOperator(op);
    bool isComp = isComparisonOperator(op);
    EXPECT_FALSE(isArith && isComp) << "Operator " << operatorToString(op)
                                    << " is both arithmetic and comparison";
    EXPECT_TRUE(isArith || isComp) << "Operator " << operatorToString(op)
                                   << " is neither arithmetic nor comparison";
  }
}
