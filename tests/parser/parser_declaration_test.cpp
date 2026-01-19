#include "parser_test_helper.hpp"

// ============== Variable Declaration Tests ==============

TEST(ParserTest, SimpleVariableDeclaration) {
  auto block = parseOrFail("let x = 5");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block.get());
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id->name, "x");
  EXPECT_EQ(varDecl->type, nullptr); // type inferred, not yet set

  // Check assignment expression is an integer
  auto* intExpr = dynamic_cast<NInteger*>(varDecl->assignmentExpr.get());
  ASSERT_NE(intExpr, nullptr);
  EXPECT_EQ(intExpr->value, 5);
}

TEST(ParserTest, TypedVariableDeclaration) {
  auto block = parseOrFail("let pi : double = 3.14");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block.get());
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id->name, "pi");
  ASSERT_NE(varDecl->type, nullptr);
  EXPECT_EQ(varDecl->type->name, "double");

  auto* doubleExpr = dynamic_cast<NDouble*>(varDecl->assignmentExpr.get());
  ASSERT_NE(doubleExpr, nullptr);
  EXPECT_DOUBLE_EQ(doubleExpr->value, 3.14);
}

TEST(ParserTest, VariableWithIdentifierExpression) {
  auto block = parseOrFail("let y = x");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block.get());
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id->name, "y");

  auto* identExpr = dynamic_cast<NIdentifier*>(varDecl->assignmentExpr.get());
  ASSERT_NE(identExpr, nullptr);
  EXPECT_EQ(identExpr->name, "x");
}

// ============== Mutable Variable Declaration Tests ==============

TEST(ParserTest, MutableVariableDeclaration) {
  auto block = parseOrFail("let x = mut 5");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block.get());
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id->name, "x");
  EXPECT_TRUE(varDecl->isMutable);

  // The expression is wrapped in NMutRefExpression by the parser
  auto* mutRefExpr = dynamic_cast<NMutRefExpression*>(varDecl->assignmentExpr.get());
  ASSERT_NE(mutRefExpr, nullptr);
  auto* intExpr = dynamic_cast<NInteger*>(mutRefExpr->expr.get());
  ASSERT_NE(intExpr, nullptr);
  EXPECT_EQ(intExpr->value, 5);
}

TEST(ParserTest, MutableTypedVariableDeclaration) {
  auto block = parseOrFail("let counter : int = mut 0");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block.get());
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id->name, "counter");
  EXPECT_TRUE(varDecl->isMutable);
  ASSERT_NE(varDecl->type, nullptr);
  EXPECT_EQ(varDecl->type->name, "int");
}

TEST(ParserTest, ImmutableVariableDeclaration) {
  auto block = parseOrFail("let x = 5");
  ASSERT_NE(block, nullptr);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block.get());
  ASSERT_NE(varDecl, nullptr);
  EXPECT_FALSE(varDecl->isMutable);
}

// ============== Function Declaration Tests ==============

TEST(ParserTest, SimpleFunctionDeclaration) {
  auto block = parseOrFail("let square(n: int): int = n * n");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block.get());
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id->name, "square");
  ASSERT_NE(funcDecl->type, nullptr);
  EXPECT_EQ(funcDecl->type->name, "int");
  ASSERT_EQ(funcDecl->arguments.size(), 1);
  EXPECT_EQ(funcDecl->arguments[0]->id->name, "n");
  ASSERT_NE(funcDecl->arguments[0]->type, nullptr);
  EXPECT_EQ(funcDecl->arguments[0]->type->name, "int");
}

TEST(ParserTest, FunctionWithMultipleArgs) {
  auto block = parseOrFail("let add(x: int, y: int): int = x + y");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block.get());
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id->name, "add");
  ASSERT_NE(funcDecl->type, nullptr);
  EXPECT_EQ(funcDecl->type->name, "int");
  ASSERT_EQ(funcDecl->arguments.size(), 2);
  EXPECT_EQ(funcDecl->arguments[0]->id->name, "x");
  EXPECT_EQ(funcDecl->arguments[1]->id->name, "y");
}

TEST(ParserTest, FunctionWithInferredReturnType) {
  auto block = parseOrFail("let double(x: int) = x + x");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block.get());
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id->name, "double");
  EXPECT_EQ(funcDecl->type, nullptr); // return type inferred, not yet set
}
