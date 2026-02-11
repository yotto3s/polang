#include "parser_test_helper.hpp"

// ============== Variable Declaration Tests ==============

TEST(ParserTest, SimpleVariableDeclaration) {
  auto block = parseOrFail("x = 5");
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
  auto block = parseOrFail("pi : double\npi = 3.14");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 2);

  // First statement: type signature
  auto* typeSig =
      dynamic_cast<NTypeSignature*>(block->statements[0].get());
  ASSERT_NE(typeSig, nullptr);
  EXPECT_EQ(typeSig->id->name, "pi");
  EXPECT_EQ(typeSig->typeExpr->getTypeName(), "double");

  // Second statement: variable declaration (type applied by type checker)
  auto* varDecl =
      dynamic_cast<NVariableDeclaration*>(block->statements[1].get());
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id->name, "pi");
  EXPECT_EQ(varDecl->type, nullptr); // type not yet applied by parser

  auto* doubleExpr = dynamic_cast<NDouble*>(varDecl->assignmentExpr.get());
  ASSERT_NE(doubleExpr, nullptr);
  EXPECT_DOUBLE_EQ(doubleExpr->value, 3.14);
}

TEST(ParserTest, VariableWithIdentifierExpression) {
  auto block = parseOrFail("y = x");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block.get());
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id->name, "y");

  auto* identExpr = dynamic_cast<NIdentifier*>(varDecl->assignmentExpr.get());
  ASSERT_NE(identExpr, nullptr);
  EXPECT_EQ(identExpr->name, "x");
}

// ============== Function Declaration Tests ==============

TEST(ParserTest, SimpleFunctionDeclaration) {
  auto block = parseOrFail("square : int -> int\nsquare(n) = n * n");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 2);

  // First statement: type signature
  auto* typeSig =
      dynamic_cast<NTypeSignature*>(block->statements[0].get());
  ASSERT_NE(typeSig, nullptr);
  EXPECT_EQ(typeSig->id->name, "square");
  EXPECT_EQ(typeSig->typeExpr->getTypeName(), "int -> int");

  // Second statement: function declaration (types applied by type checker)
  auto* funcDecl =
      dynamic_cast<NFunctionDeclaration*>(block->statements[1].get());
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id->name, "square");
  EXPECT_EQ(funcDecl->type, nullptr); // return type applied by type checker
  ASSERT_EQ(funcDecl->arguments.size(), 1);
  EXPECT_EQ(funcDecl->arguments[0]->id->name, "n");
  EXPECT_EQ(funcDecl->arguments[0]->type, nullptr); // param type applied by type checker
}

TEST(ParserTest, FunctionWithMultipleArgs) {
  auto block = parseOrFail("add : int * int -> int\nadd(x, y) = x + y");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 2);

  // First statement: type signature
  auto* typeSig =
      dynamic_cast<NTypeSignature*>(block->statements[0].get());
  ASSERT_NE(typeSig, nullptr);
  EXPECT_EQ(typeSig->id->name, "add");
  EXPECT_EQ(typeSig->typeExpr->getTypeName(), "int * int -> int");

  // Second statement: function declaration
  auto* funcDecl =
      dynamic_cast<NFunctionDeclaration*>(block->statements[1].get());
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id->name, "add");
  EXPECT_EQ(funcDecl->type, nullptr); // return type applied by type checker
  ASSERT_EQ(funcDecl->arguments.size(), 2);
  EXPECT_EQ(funcDecl->arguments[0]->id->name, "x");
  EXPECT_EQ(funcDecl->arguments[1]->id->name, "y");
}

TEST(ParserTest, FunctionWithInferredReturnType) {
  auto block = parseOrFail("double(x) = x + x");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block.get());
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id->name, "double");
  EXPECT_EQ(funcDecl->type, nullptr); // return type inferred, not yet set
}
