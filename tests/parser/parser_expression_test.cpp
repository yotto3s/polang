#include "parser_test_helper.hpp"

// ============== Literal Expression Tests ==============

TEST(ParserTest, IntegerExpression) {
  NBlock* block = parseOrFail("42");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto* intExpr = dynamic_cast<const NInteger*>(&exprStmt->expression);
  ASSERT_NE(intExpr, nullptr);
  EXPECT_EQ(intExpr->value, 42);
}

TEST(ParserTest, DoubleExpression) {
  NBlock* block = parseOrFail("3.14159");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto* doubleExpr = dynamic_cast<const NDouble*>(&exprStmt->expression);
  ASSERT_NE(doubleExpr, nullptr);
  EXPECT_DOUBLE_EQ(doubleExpr->value, 3.14159);
}

TEST(ParserTest, IdentifierExpression) {
  NBlock* block = parseOrFail("myVar");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto* identExpr = dynamic_cast<const NIdentifier*>(&exprStmt->expression);
  ASSERT_NE(identExpr, nullptr);
  EXPECT_EQ(identExpr->name, "myVar");
}

// ============== Binary Operator Tests ==============

TEST(ParserTest, AdditionExpression) {
  NBlock* block = parseOrFail("1 + 2");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto* binOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TPLUS);

  auto* lhs = dynamic_cast<const NInteger*>(&binOp->lhs);
  auto* rhs = dynamic_cast<const NInteger*>(&binOp->rhs);
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(lhs->value, 1);
  EXPECT_EQ(rhs->value, 2);
}

TEST(ParserTest, SubtractionExpression) {
  NBlock* block = parseOrFail("5 - 3");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* binOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TMINUS);
}

TEST(ParserTest, MultiplicationExpression) {
  NBlock* block = parseOrFail("4 * 5");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* binOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TMUL);
}

TEST(ParserTest, DivisionExpression) {
  NBlock* block = parseOrFail("10 / 2");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* binOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TDIV);
}

// ============== Operator Precedence Tests ==============

TEST(ParserTest, MulHigherThanAdd) {
  // 1 + 2 * 3 should parse as 1 + (2 * 3)
  NBlock* block = parseOrFail("1 + 2 * 3");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* addOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, TPLUS);

  // LHS should be 1
  auto* lhs = dynamic_cast<const NInteger*>(&addOp->lhs);
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->value, 1);

  // RHS should be 2 * 3
  auto* mulOp = dynamic_cast<const NBinaryOperator*>(&addOp->rhs);
  ASSERT_NE(mulOp, nullptr);
  EXPECT_EQ(mulOp->op, TMUL);
}

TEST(ParserTest, ParenthesesOverridePrecedence) {
  // (1 + 2) * 3 should parse as (1 + 2) * 3
  NBlock* block = parseOrFail("(1 + 2) * 3");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* mulOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(mulOp, nullptr);
  EXPECT_EQ(mulOp->op, TMUL);

  // LHS should be 1 + 2
  auto* addOp = dynamic_cast<const NBinaryOperator*>(&mulOp->lhs);
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, TPLUS);
}

// ============== Comparison Operator Tests ==============

TEST(ParserTest, EqualComparison) {
  NBlock* block = parseOrFail("a == b");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* binOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TCEQ);
}

TEST(ParserTest, NotEqualComparison) {
  NBlock* block = parseOrFail("a != b");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* binOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TCNE);
}

TEST(ParserTest, LessThanComparison) {
  NBlock* block = parseOrFail("a < b");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* binOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TCLT);
}

TEST(ParserTest, GreaterThanComparison) {
  NBlock* block = parseOrFail("a > b");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* binOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TCGT);
}

// ============== Function Call Tests ==============

TEST(ParserTest, FunctionCallNoArgs) {
  NBlock* block = parseOrFail("foo()");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* call = dynamic_cast<const NMethodCall*>(&exprStmt->expression);
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id.name, "foo");
  EXPECT_EQ(call->arguments.size(), 0);
}

TEST(ParserTest, FunctionCallOneArg) {
  NBlock* block = parseOrFail("square(5)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* call = dynamic_cast<const NMethodCall*>(&exprStmt->expression);
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id.name, "square");
  ASSERT_EQ(call->arguments.size(), 1);

  auto* arg = dynamic_cast<NInteger*>(call->arguments[0]);
  ASSERT_NE(arg, nullptr);
  EXPECT_EQ(arg->value, 5);
}

TEST(ParserTest, FunctionCallMultipleArgs) {
  NBlock* block = parseOrFail("add(1, 2, 3)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* call = dynamic_cast<const NMethodCall*>(&exprStmt->expression);
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id.name, "add");
  ASSERT_EQ(call->arguments.size(), 3);
}

// ============== Assignment Tests ==============

TEST(ParserTest, SimpleAssignment) {
  NBlock* block = parseOrFail("x <- 10");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* assign = dynamic_cast<const NAssignment*>(&exprStmt->expression);
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->lhs.name, "x");

  auto* rhs = dynamic_cast<const NInteger*>(&assign->rhs);
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->value, 10);
}

TEST(ParserTest, AssignmentWithExpression) {
  NBlock* block = parseOrFail("x <- 1 + 2");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* assign = dynamic_cast<const NAssignment*>(&exprStmt->expression);
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->lhs.name, "x");

  auto* binOp = dynamic_cast<const NBinaryOperator*>(&assign->rhs);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TPLUS);
}

TEST(ParserTest, ChainedAssignment) {
  // x <- y <- 5 should parse as x <- (y <- 5) due to right-associativity
  NBlock* block = parseOrFail("x <- y <- 5");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* outerAssign = dynamic_cast<const NAssignment*>(&exprStmt->expression);
  ASSERT_NE(outerAssign, nullptr);
  EXPECT_EQ(outerAssign->lhs.name, "x");

  auto* innerAssign = dynamic_cast<const NAssignment*>(&outerAssign->rhs);
  ASSERT_NE(innerAssign, nullptr);
  EXPECT_EQ(innerAssign->lhs.name, "y");
}

// ============== Complex Expression Tests ==============

TEST(ParserTest, ComplexArithmeticExpression) {
  NBlock* block = parseOrFail("(a + b) * (c - d) / e");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  // Top level should be division
  auto* divOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(divOp, nullptr);
  EXPECT_EQ(divOp->op, TDIV);
}

TEST(ParserTest, FunctionCallInExpression) {
  NBlock* block = parseOrFail("foo(1) + bar(2)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* addOp = dynamic_cast<const NBinaryOperator*>(&exprStmt->expression);
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, TPLUS);

  auto* lhs = dynamic_cast<const NMethodCall*>(&addOp->lhs);
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->id.name, "foo");

  auto* rhs = dynamic_cast<const NMethodCall*>(&addOp->rhs);
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->id.name, "bar");
}
