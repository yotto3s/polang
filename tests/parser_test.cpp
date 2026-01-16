// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
#include <string>

// clang-format off
// Parser headers (includes LLVM via node.hpp)
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp - provides token constants
#include "parser/parser_api.hpp"
// clang-format on

// ============== Helper Functions ==============

// Parse and return root block, fails test if null
NBlock *parseOrFail(const std::string &source) {
  NBlock *block = polang_parse(source);
  EXPECT_NE(block, nullptr) << "Failed to parse: " << source;
  return block;
}

// Get first statement as specific type
template <typename T> T *getFirstStatement(NBlock *block) {
  EXPECT_FALSE(block->statements.empty());
  if (block->statements.empty())
    return nullptr;
  T *stmt = dynamic_cast<T *>(block->statements[0]);
  EXPECT_NE(stmt, nullptr) << "First statement is not of expected type";
  return stmt;
}

// ============== Variable Declaration Tests ==============

TEST(ParserTest, SimpleVariableDeclaration) {
  NBlock *block = parseOrFail("let x = 5");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "x");
  EXPECT_EQ(varDecl->type.name, "int"); // default type

  // Check assignment expression is an integer
  auto *intExpr = dynamic_cast<NInteger *>(varDecl->assignmentExpr);
  ASSERT_NE(intExpr, nullptr);
  EXPECT_EQ(intExpr->value, 5);
}

TEST(ParserTest, TypedVariableDeclaration) {
  NBlock *block = parseOrFail("let pi : double = 3.14");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "pi");
  EXPECT_EQ(varDecl->type.name, "double");

  auto *doubleExpr = dynamic_cast<NDouble *>(varDecl->assignmentExpr);
  ASSERT_NE(doubleExpr, nullptr);
  EXPECT_DOUBLE_EQ(doubleExpr->value, 3.14);
}

TEST(ParserTest, VariableWithIdentifierExpression) {
  NBlock *block = parseOrFail("let y = x");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "y");

  auto *identExpr = dynamic_cast<NIdentifier *>(varDecl->assignmentExpr);
  ASSERT_NE(identExpr, nullptr);
  EXPECT_EQ(identExpr->name, "x");
}

// ============== Function Declaration Tests ==============

TEST(ParserTest, SimpleFunctionDeclaration) {
  NBlock *block = parseOrFail("let square (n : int) : int = n * n");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "square");
  EXPECT_EQ(funcDecl->type.name, "int");
  ASSERT_EQ(funcDecl->arguments.size(), 1);
  EXPECT_EQ(funcDecl->arguments[0]->id.name, "n");
  EXPECT_EQ(funcDecl->arguments[0]->type.name, "int");
}

TEST(ParserTest, FunctionWithMultipleArgs) {
  NBlock *block = parseOrFail("let add (x : int) (y : int) : int = x + y");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "add");
  EXPECT_EQ(funcDecl->type.name, "int");
  ASSERT_EQ(funcDecl->arguments.size(), 2);
  EXPECT_EQ(funcDecl->arguments[0]->id.name, "x");
  EXPECT_EQ(funcDecl->arguments[1]->id.name, "y");
}

TEST(ParserTest, FunctionWithInferredReturnType) {
  NBlock *block = parseOrFail("let double (x : int) = x + x");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "double");
  EXPECT_EQ(funcDecl->type.name, "int"); // inferred
}

// ============== Expression Statement Tests ==============

TEST(ParserTest, IntegerExpression) {
  NBlock *block = parseOrFail("42");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto *intExpr = dynamic_cast<NInteger *>(&exprStmt->expression);
  ASSERT_NE(intExpr, nullptr);
  EXPECT_EQ(intExpr->value, 42);
}

TEST(ParserTest, DoubleExpression) {
  NBlock *block = parseOrFail("3.14159");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto *doubleExpr = dynamic_cast<NDouble *>(&exprStmt->expression);
  ASSERT_NE(doubleExpr, nullptr);
  EXPECT_DOUBLE_EQ(doubleExpr->value, 3.14159);
}

TEST(ParserTest, IdentifierExpression) {
  NBlock *block = parseOrFail("myVar");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto *identExpr = dynamic_cast<NIdentifier *>(&exprStmt->expression);
  ASSERT_NE(identExpr, nullptr);
  EXPECT_EQ(identExpr->name, "myVar");
}

// ============== Binary Operator Tests ==============

TEST(ParserTest, AdditionExpression) {
  NBlock *block = parseOrFail("1 + 2");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto *binOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TPLUS);

  auto *lhs = dynamic_cast<NInteger *>(&binOp->lhs);
  auto *rhs = dynamic_cast<NInteger *>(&binOp->rhs);
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(lhs->value, 1);
  EXPECT_EQ(rhs->value, 2);
}

TEST(ParserTest, SubtractionExpression) {
  NBlock *block = parseOrFail("5 - 3");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *binOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TMINUS);
}

TEST(ParserTest, MultiplicationExpression) {
  NBlock *block = parseOrFail("4 * 5");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *binOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TMUL);
}

TEST(ParserTest, DivisionExpression) {
  NBlock *block = parseOrFail("10 / 2");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *binOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TDIV);
}

// ============== Operator Precedence Tests ==============

TEST(ParserTest, MulHigherThanAdd) {
  // 1 + 2 * 3 should parse as 1 + (2 * 3)
  NBlock *block = parseOrFail("1 + 2 * 3");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *addOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, TPLUS);

  // LHS should be 1
  auto *lhs = dynamic_cast<NInteger *>(&addOp->lhs);
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->value, 1);

  // RHS should be 2 * 3
  auto *mulOp = dynamic_cast<NBinaryOperator *>(&addOp->rhs);
  ASSERT_NE(mulOp, nullptr);
  EXPECT_EQ(mulOp->op, TMUL);
}

TEST(ParserTest, ParenthesesOverridePrecedence) {
  // (1 + 2) * 3 should parse as (1 + 2) * 3
  NBlock *block = parseOrFail("(1 + 2) * 3");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *mulOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(mulOp, nullptr);
  EXPECT_EQ(mulOp->op, TMUL);

  // LHS should be 1 + 2
  auto *addOp = dynamic_cast<NBinaryOperator *>(&mulOp->lhs);
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, TPLUS);
}

// ============== Comparison Operator Tests ==============

TEST(ParserTest, EqualComparison) {
  NBlock *block = parseOrFail("a == b");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *binOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TCEQ);
}

TEST(ParserTest, NotEqualComparison) {
  NBlock *block = parseOrFail("a != b");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *binOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TCNE);
}

TEST(ParserTest, LessThanComparison) {
  NBlock *block = parseOrFail("a < b");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *binOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TCLT);
}

TEST(ParserTest, GreaterThanComparison) {
  NBlock *block = parseOrFail("a > b");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *binOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(binOp, nullptr);
  EXPECT_EQ(binOp->op, TCGT);
}

// ============== Function Call Tests ==============

TEST(ParserTest, FunctionCallNoArgs) {
  NBlock *block = parseOrFail("foo()");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *call = dynamic_cast<NMethodCall *>(&exprStmt->expression);
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id.name, "foo");
  EXPECT_EQ(call->arguments.size(), 0);
}

TEST(ParserTest, FunctionCallOneArg) {
  NBlock *block = parseOrFail("square(5)");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *call = dynamic_cast<NMethodCall *>(&exprStmt->expression);
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id.name, "square");
  ASSERT_EQ(call->arguments.size(), 1);

  auto *arg = dynamic_cast<NInteger *>(call->arguments[0]);
  ASSERT_NE(arg, nullptr);
  EXPECT_EQ(arg->value, 5);
}

TEST(ParserTest, FunctionCallMultipleArgs) {
  NBlock *block = parseOrFail("add(1, 2, 3)");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *call = dynamic_cast<NMethodCall *>(&exprStmt->expression);
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id.name, "add");
  ASSERT_EQ(call->arguments.size(), 3);
}

// ============== Assignment Tests ==============

TEST(ParserTest, SimpleAssignment) {
  NBlock *block = parseOrFail("x = 10");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *assign = dynamic_cast<NAssignment *>(&exprStmt->expression);
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->lhs.name, "x");

  auto *rhs = dynamic_cast<NInteger *>(&assign->rhs);
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->value, 10);
}

// ============== If Expression Tests ==============

TEST(ParserTest, SimpleIfExpression) {
  NBlock *block = parseOrFail("if 1 then 2 else 3");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *ifExpr = dynamic_cast<NIfExpression *>(&exprStmt->expression);
  ASSERT_NE(ifExpr, nullptr);

  auto *cond = dynamic_cast<NInteger *>(&ifExpr->condition);
  ASSERT_NE(cond, nullptr);
  EXPECT_EQ(cond->value, 1);

  auto *thenExpr = dynamic_cast<NInteger *>(&ifExpr->thenExpr);
  ASSERT_NE(thenExpr, nullptr);
  EXPECT_EQ(thenExpr->value, 2);

  auto *elseExpr = dynamic_cast<NInteger *>(&ifExpr->elseExpr);
  ASSERT_NE(elseExpr, nullptr);
  EXPECT_EQ(elseExpr->value, 3);
}

TEST(ParserTest, IfExpressionWithComparison) {
  NBlock *block = parseOrFail("if x > 0 then 1 else 0");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *ifExpr = dynamic_cast<NIfExpression *>(&exprStmt->expression);
  ASSERT_NE(ifExpr, nullptr);

  auto *cond = dynamic_cast<NBinaryOperator *>(&ifExpr->condition);
  ASSERT_NE(cond, nullptr);
  EXPECT_EQ(cond->op, TCGT);
}

TEST(ParserTest, NestedIfExpression) {
  NBlock *block = parseOrFail("if a then if b then 1 else 2 else 3");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *outerIf = dynamic_cast<NIfExpression *>(&exprStmt->expression);
  ASSERT_NE(outerIf, nullptr);

  // The then branch should be another if expression
  auto *innerIf = dynamic_cast<NIfExpression *>(&outerIf->thenExpr);
  ASSERT_NE(innerIf, nullptr);

  // The outer else should be 3
  auto *elseExpr = dynamic_cast<NInteger *>(&outerIf->elseExpr);
  ASSERT_NE(elseExpr, nullptr);
  EXPECT_EQ(elseExpr->value, 3);
}

TEST(ParserTest, IfExpressionInVariableDeclaration) {
  NBlock *block = parseOrFail("let x = if a then 1 else 0");
  ASSERT_NE(block, nullptr);

  auto *varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "x");

  auto *ifExpr = dynamic_cast<NIfExpression *>(varDecl->assignmentExpr);
  ASSERT_NE(ifExpr, nullptr);
}

TEST(ParserTest, IfExpressionInFunctionBody) {
  NBlock *block =
      parseOrFail("let max (a : int) (b : int) : int = if a > b then a else b");
  ASSERT_NE(block, nullptr);

  auto *funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "max");

  // Function body is a block with one expression statement
  ASSERT_EQ(funcDecl->block.statements.size(), 1);
  auto *bodyStmt =
      dynamic_cast<NExpressionStatement *>(funcDecl->block.statements[0]);
  ASSERT_NE(bodyStmt, nullptr);

  auto *ifExpr = dynamic_cast<NIfExpression *>(&bodyStmt->expression);
  ASSERT_NE(ifExpr, nullptr);
}

// ============== Multiple Statements Tests ==============

TEST(ParserTest, MultipleStatements) {
  NBlock *block = parseOrFail("let x = 1\nlet y = 2\nx + y");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 3);

  auto *varDecl1 = dynamic_cast<NVariableDeclaration *>(block->statements[0]);
  ASSERT_NE(varDecl1, nullptr);
  EXPECT_EQ(varDecl1->id.name, "x");

  auto *varDecl2 = dynamic_cast<NVariableDeclaration *>(block->statements[1]);
  ASSERT_NE(varDecl2, nullptr);
  EXPECT_EQ(varDecl2->id.name, "y");

  auto *exprStmt = dynamic_cast<NExpressionStatement *>(block->statements[2]);
  ASSERT_NE(exprStmt, nullptr);
}

TEST(ParserTest, FunctionAndCall) {
  NBlock *block = parseOrFail("let inc (x : int) : int = x + 1\ninc(5)");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 2);

  auto *funcDecl = dynamic_cast<NFunctionDeclaration *>(block->statements[0]);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "inc");

  auto *exprStmt = dynamic_cast<NExpressionStatement *>(block->statements[1]);
  ASSERT_NE(exprStmt, nullptr);
  auto *call = dynamic_cast<NMethodCall *>(&exprStmt->expression);
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id.name, "inc");
}

// ============== Complex Expression Tests ==============

TEST(ParserTest, ComplexArithmeticExpression) {
  NBlock *block = parseOrFail("(a + b) * (c - d) / e");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  // Top level should be division
  auto *divOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(divOp, nullptr);
  EXPECT_EQ(divOp->op, TDIV);
}

TEST(ParserTest, FunctionCallInExpression) {
  NBlock *block = parseOrFail("foo(1) + bar(2)");
  ASSERT_NE(block, nullptr);

  auto *exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto *addOp = dynamic_cast<NBinaryOperator *>(&exprStmt->expression);
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, TPLUS);

  auto *lhs = dynamic_cast<NMethodCall *>(&addOp->lhs);
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->id.name, "foo");

  auto *rhs = dynamic_cast<NMethodCall *>(&addOp->rhs);
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->id.name, "bar");
}

// ============== Edge Cases ==============

TEST(ParserTest, EmptyInput) {
  // Empty input should return an empty block
  NBlock *block = polang_parse("");
  // Note: Depending on implementation, this might be nullptr or empty block
  // Just check it doesn't crash
  SUCCEED();
}

TEST(ParserTest, WhitespaceOnlyInput) {
  NBlock *block = polang_parse("   \n\t  ");
  SUCCEED();
}
