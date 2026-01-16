#include <gtest/gtest.h>

// clang-format off
#include "parser/operator_utils.hpp"
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison union types
// clang-format on

using namespace polang;

// ============== operatorToString Tests ==============

TEST(OperatorUtilsTest, OperatorToStringPlus) {
  EXPECT_EQ(operatorToString(TPLUS), "+");
}

TEST(OperatorUtilsTest, OperatorToStringMinus) {
  EXPECT_EQ(operatorToString(TMINUS), "-");
}

TEST(OperatorUtilsTest, OperatorToStringMul) {
  EXPECT_EQ(operatorToString(TMUL), "*");
}

TEST(OperatorUtilsTest, OperatorToStringDiv) {
  EXPECT_EQ(operatorToString(TDIV), "/");
}

TEST(OperatorUtilsTest, OperatorToStringEqual) {
  EXPECT_EQ(operatorToString(TCEQ), "==");
}

TEST(OperatorUtilsTest, OperatorToStringNotEqual) {
  EXPECT_EQ(operatorToString(TCNE), "!=");
}

TEST(OperatorUtilsTest, OperatorToStringLessThan) {
  EXPECT_EQ(operatorToString(TCLT), "<");
}

TEST(OperatorUtilsTest, OperatorToStringLessEqual) {
  EXPECT_EQ(operatorToString(TCLE), "<=");
}

TEST(OperatorUtilsTest, OperatorToStringGreaterThan) {
  EXPECT_EQ(operatorToString(TCGT), ">");
}

TEST(OperatorUtilsTest, OperatorToStringGreaterEqual) {
  EXPECT_EQ(operatorToString(TCGE), ">=");
}

TEST(OperatorUtilsTest, OperatorToStringUnknown) {
  // Unknown operator should return "?"
  EXPECT_EQ(operatorToString(-1), "?");
  EXPECT_EQ(operatorToString(9999), "?");
}

// ============== isArithmeticOperator Tests ==============

TEST(OperatorUtilsTest, IsArithmeticOperatorPlus) {
  EXPECT_TRUE(isArithmeticOperator(TPLUS));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorMinus) {
  EXPECT_TRUE(isArithmeticOperator(TMINUS));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorMul) {
  EXPECT_TRUE(isArithmeticOperator(TMUL));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorDiv) {
  EXPECT_TRUE(isArithmeticOperator(TDIV));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorComparison) {
  // Comparison operators are not arithmetic
  EXPECT_FALSE(isArithmeticOperator(TCEQ));
  EXPECT_FALSE(isArithmeticOperator(TCNE));
  EXPECT_FALSE(isArithmeticOperator(TCLT));
  EXPECT_FALSE(isArithmeticOperator(TCLE));
  EXPECT_FALSE(isArithmeticOperator(TCGT));
  EXPECT_FALSE(isArithmeticOperator(TCGE));
}

TEST(OperatorUtilsTest, IsArithmeticOperatorUnknown) {
  EXPECT_FALSE(isArithmeticOperator(-1));
  EXPECT_FALSE(isArithmeticOperator(0));
}

// ============== isComparisonOperator Tests ==============

TEST(OperatorUtilsTest, IsComparisonOperatorEqual) {
  EXPECT_TRUE(isComparisonOperator(TCEQ));
}

TEST(OperatorUtilsTest, IsComparisonOperatorNotEqual) {
  EXPECT_TRUE(isComparisonOperator(TCNE));
}

TEST(OperatorUtilsTest, IsComparisonOperatorLessThan) {
  EXPECT_TRUE(isComparisonOperator(TCLT));
}

TEST(OperatorUtilsTest, IsComparisonOperatorLessEqual) {
  EXPECT_TRUE(isComparisonOperator(TCLE));
}

TEST(OperatorUtilsTest, IsComparisonOperatorGreaterThan) {
  EXPECT_TRUE(isComparisonOperator(TCGT));
}

TEST(OperatorUtilsTest, IsComparisonOperatorGreaterEqual) {
  EXPECT_TRUE(isComparisonOperator(TCGE));
}

TEST(OperatorUtilsTest, IsComparisonOperatorArithmetic) {
  // Arithmetic operators are not comparison
  EXPECT_FALSE(isComparisonOperator(TPLUS));
  EXPECT_FALSE(isComparisonOperator(TMINUS));
  EXPECT_FALSE(isComparisonOperator(TMUL));
  EXPECT_FALSE(isComparisonOperator(TDIV));
}

TEST(OperatorUtilsTest, IsComparisonOperatorUnknown) {
  EXPECT_FALSE(isComparisonOperator(-1));
  EXPECT_FALSE(isComparisonOperator(0));
}

// ============== Mutual Exclusion Tests ==============

TEST(OperatorUtilsTest, ArithmeticAndComparisonMutuallyExclusive) {
  // No operator should be both arithmetic and comparison
  const int operators[] = {TPLUS, TMINUS, TMUL, TDIV, TCEQ,
                           TCNE,  TCLT,   TCLE, TCGT, TCGE};

  for (int op : operators) {
    bool isArith = isArithmeticOperator(op);
    bool isComp = isComparisonOperator(op);
    EXPECT_FALSE(isArith && isComp) << "Operator " << operatorToString(op)
                                    << " is both arithmetic and comparison";
    EXPECT_TRUE(isArith || isComp) << "Operator " << operatorToString(op)
                                   << " is neither arithmetic nor comparison";
  }
}
