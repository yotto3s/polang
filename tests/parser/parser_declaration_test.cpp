#include "parser_test_helper.hpp"

// ============== Variable Declaration Tests ==============

TEST(ParserTest, SimpleVariableDeclaration) {
  NBlock* block = parseOrFail("let x = 5");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "x");
  EXPECT_EQ(varDecl->type, nullptr); // type inferred, not yet set

  // Check assignment expression is an integer
  auto* intExpr = dynamic_cast<NInteger*>(varDecl->assignmentExpr);
  ASSERT_NE(intExpr, nullptr);
  EXPECT_EQ(intExpr->value, 5);
}

TEST(ParserTest, TypedVariableDeclaration) {
  NBlock* block = parseOrFail("let pi : double = 3.14");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "pi");
  ASSERT_NE(varDecl->type, nullptr);
  EXPECT_EQ(varDecl->type->name, "double");

  auto* doubleExpr = dynamic_cast<NDouble*>(varDecl->assignmentExpr);
  ASSERT_NE(doubleExpr, nullptr);
  EXPECT_DOUBLE_EQ(doubleExpr->value, 3.14);
}

TEST(ParserTest, VariableWithIdentifierExpression) {
  NBlock* block = parseOrFail("let y = x");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "y");

  auto* identExpr = dynamic_cast<NIdentifier*>(varDecl->assignmentExpr);
  ASSERT_NE(identExpr, nullptr);
  EXPECT_EQ(identExpr->name, "x");
}

// ============== Function Declaration Tests ==============

TEST(ParserTest, SimpleFunctionDeclaration) {
  NBlock* block = parseOrFail("let square (n : int) : int = n * n");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "square");
  ASSERT_NE(funcDecl->type, nullptr);
  EXPECT_EQ(funcDecl->type->name, "int");
  ASSERT_EQ(funcDecl->arguments.size(), 1);
  EXPECT_EQ(funcDecl->arguments[0]->id.name, "n");
  ASSERT_NE(funcDecl->arguments[0]->type, nullptr);
  EXPECT_EQ(funcDecl->arguments[0]->type->name, "int");
}

TEST(ParserTest, FunctionWithMultipleArgs) {
  NBlock* block = parseOrFail("let add (x : int) (y : int) : int = x + y");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "add");
  ASSERT_NE(funcDecl->type, nullptr);
  EXPECT_EQ(funcDecl->type->name, "int");
  ASSERT_EQ(funcDecl->arguments.size(), 2);
  EXPECT_EQ(funcDecl->arguments[0]->id.name, "x");
  EXPECT_EQ(funcDecl->arguments[1]->id.name, "y");
}

TEST(ParserTest, FunctionWithInferredReturnType) {
  NBlock* block = parseOrFail("let double (x : int) = x + x");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "double");
  EXPECT_EQ(funcDecl->type, nullptr); // return type inferred, not yet set
}
