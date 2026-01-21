#include "parser_test_helper.hpp"

// ============== Literal Expression Tests ==============

TEST(ParserTest, IntegerExpression) {
  auto block = parseOrFail("42");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  ASSERT_NE(exprStmt, nullptr);

  auto* intExpr = dynamic_cast<const NInteger*>(exprStmt->expression.get());
  ASSERT_NE(intExpr, nullptr);
  EXPECT_EQ(intExpr->value, 42);
}

TEST(ParserTest, DoubleExpression) {
  auto block = parseOrFail("3.14159");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  ASSERT_NE(exprStmt, nullptr);

  auto* doubleExpr = dynamic_cast<const NDouble*>(exprStmt->expression.get());
  ASSERT_NE(doubleExpr, nullptr);
  EXPECT_DOUBLE_EQ(doubleExpr->value, 3.14159);
}

TEST(ParserTest, IdentifierExpression) {
  auto block = parseOrFail("myVar");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  ASSERT_NE(exprStmt, nullptr);

  auto* identExpr = dynamic_cast<const NIdentifier*>(exprStmt->expression.get());
  ASSERT_NE(identExpr, nullptr);
  EXPECT_EQ(identExpr->name, "myVar");
}

// ============== Binary Operator Tests ==============

TEST(ParserTest, AdditionExpression) {
  auto block = parseOrFail("1 + 2");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  ASSERT_NE(exprStmt, nullptr);

  auto* binOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, token::TPLUS);

  auto* lhs = dynamic_cast<const NInteger*>(binOp->lhs.get());
  auto* rhs = dynamic_cast<const NInteger*>(binOp->rhs.get());
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(lhs->value, 1);
  EXPECT_EQ(rhs->value, 2);
}

TEST(ParserTest, SubtractionExpression) {
  auto block = parseOrFail("5 - 3");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* binOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, token::TMINUS);
}

TEST(ParserTest, MultiplicationExpression) {
  auto block = parseOrFail("4 * 5");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* binOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, token::TMUL);
}

TEST(ParserTest, DivisionExpression) {
  auto block = parseOrFail("10 / 2");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* binOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, token::TDIV);
}

// ============== Operator Precedence Tests ==============

TEST(ParserTest, MulHigherThanAdd) {
  // 1 + 2 * 3 should parse as 1 + (2 * 3)
  auto block = parseOrFail("1 + 2 * 3");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* addOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, token::TPLUS);

  // LHS should be 1
  auto* lhs = dynamic_cast<const NInteger*>(addOp->lhs.get());
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->value, 1);

  // RHS should be 2 * 3
  auto* mulOp = dynamic_cast<const NBinaryOperator*>(addOp->rhs.get());
  ASSERT_NE(mulOp, nullptr);
  EXPECT_EQ(mulOp->op, token::TMUL);
}

TEST(ParserTest, ParenthesesOverridePrecedence) {
  // (1 + 2) * 3 should parse as (1 + 2) * 3
  auto block = parseOrFail("(1 + 2) * 3");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* mulOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(mulOp, nullptr);
  EXPECT_EQ(mulOp->op, token::TMUL);

  // LHS should be 1 + 2
  auto* addOp = dynamic_cast<const NBinaryOperator*>(mulOp->lhs.get());
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, token::TPLUS);
}

// ============== Comparison Operator Tests ==============

TEST(ParserTest, EqualComparison) {
  auto block = parseOrFail("a == b");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* binOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, token::TCEQ);
}

TEST(ParserTest, NotEqualComparison) {
  auto block = parseOrFail("a != b");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* binOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, token::TCNE);
}

TEST(ParserTest, LessThanComparison) {
  auto block = parseOrFail("a < b");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* binOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, token::TCLT);
}

TEST(ParserTest, GreaterThanComparison) {
  auto block = parseOrFail("a > b");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* binOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, token::TCGT);
}

// ============== Function Call Tests ==============

TEST(ParserTest, FunctionCallNoArgs) {
  auto block = parseOrFail("foo()");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* call = dynamic_cast<const NMethodCall*>(exprStmt->expression.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id->name, "foo");
  EXPECT_EQ(call->arguments.size(), 0);
}

TEST(ParserTest, FunctionCallOneArg) {
  auto block = parseOrFail("square(5)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* call = dynamic_cast<const NMethodCall*>(exprStmt->expression.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id->name, "square");
  ASSERT_EQ(call->arguments.size(), 1);

  auto* arg = dynamic_cast<NInteger*>(call->arguments[0].get());
  ASSERT_NE(arg, nullptr);
  EXPECT_EQ(arg->value, 5);
}

TEST(ParserTest, FunctionCallMultipleArgs) {
  auto block = parseOrFail("add(1, 2, 3)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* call = dynamic_cast<const NMethodCall*>(exprStmt->expression.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id->name, "add");
  ASSERT_EQ(call->arguments.size(), 3);
}

// ============== Complex Expression Tests ==============

TEST(ParserTest, ComplexArithmeticExpression) {
  auto block = parseOrFail("(a + b) * (c - d) / e");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  // Top level should be division
  auto* divOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(divOp, nullptr);
  EXPECT_EQ(divOp->op, token::TDIV);
}

TEST(ParserTest, FunctionCallInExpression) {
  auto block = parseOrFail("foo(1) + bar(2)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block.get());
  auto* addOp = dynamic_cast<const NBinaryOperator*>(exprStmt->expression.get());
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, token::TPLUS);

  auto* lhs = dynamic_cast<const NMethodCall*>(addOp->lhs.get());
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->id->name, "foo");

  auto* rhs = dynamic_cast<const NMethodCall*>(addOp->rhs.get());
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->id->name, "bar");
}
