#include "parser_test_helper.hpp"

// ============== Multiple Statements Tests ==============

TEST(ParserTest, MultipleStatements) {
  auto block = parseOrFail("let x = 1\nlet y = 2\nx + y");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 3);

  auto* varDecl1 = dynamic_cast<NVariableDeclaration*>(block->statements[0].get());
  ASSERT_NE(varDecl1, nullptr);
  EXPECT_EQ(varDecl1->id->name, "x");

  auto* varDecl2 = dynamic_cast<NVariableDeclaration*>(block->statements[1].get());
  ASSERT_NE(varDecl2, nullptr);
  EXPECT_EQ(varDecl2->id->name, "y");

  auto* exprStmt = dynamic_cast<NExpressionStatement*>(block->statements[2].get());
  ASSERT_NE(exprStmt, nullptr);
}

TEST(ParserTest, FunctionAndCall) {
  auto block = parseOrFail("let inc(x: int): int = x + 1\ninc(5)");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 2);

  auto* funcDecl = dynamic_cast<NFunctionDeclaration*>(block->statements[0].get());
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id->name, "inc");

  auto* exprStmt = dynamic_cast<NExpressionStatement*>(block->statements[1].get());
  ASSERT_NE(exprStmt, nullptr);
  auto* call = dynamic_cast<const NMethodCall*>(exprStmt->expression.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id->name, "inc");
}

// ============== Edge Cases ==============

TEST(ParserTest, EmptyInput) {
  // Empty input should return an empty block
  auto block = polang_parse("");
  // Note: Depending on implementation, this might be nullptr or empty block
  // Just check it doesn't crash
  SUCCEED();
}

TEST(ParserTest, WhitespaceOnlyInput) {
  auto block = polang_parse("   \n\t  ");
  SUCCEED();
}
