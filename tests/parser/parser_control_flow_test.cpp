#include "parser_test_helper.hpp"

// ============== If Expression Tests ==============

TEST(ParserTest, SimpleIfExpression) {
  NBlock* block = parseOrFail("if 1 then 2 else 3");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* ifExpr = dynamic_cast<const NIfExpression*>(&exprStmt->expression);
  ASSERT_NE(ifExpr, nullptr);

  auto* cond = dynamic_cast<const NInteger*>(&ifExpr->condition);
  ASSERT_NE(cond, nullptr);
  EXPECT_EQ(cond->value, 1);

  auto* thenExpr = dynamic_cast<const NInteger*>(&ifExpr->thenExpr);
  ASSERT_NE(thenExpr, nullptr);
  EXPECT_EQ(thenExpr->value, 2);

  auto* elseExpr = dynamic_cast<const NInteger*>(&ifExpr->elseExpr);
  ASSERT_NE(elseExpr, nullptr);
  EXPECT_EQ(elseExpr->value, 3);
}

TEST(ParserTest, IfExpressionWithComparison) {
  NBlock* block = parseOrFail("if x > 0 then 1 else 0");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* ifExpr = dynamic_cast<const NIfExpression*>(&exprStmt->expression);
  ASSERT_NE(ifExpr, nullptr);

  auto* cond = dynamic_cast<const NBinaryOperator*>(&ifExpr->condition);
  ASSERT_NE(cond, nullptr);
  EXPECT_EQ(cond->op, TCGT);
}

TEST(ParserTest, NestedIfExpression) {
  NBlock* block = parseOrFail("if a then if b then 1 else 2 else 3");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* outerIf = dynamic_cast<const NIfExpression*>(&exprStmt->expression);
  ASSERT_NE(outerIf, nullptr);

  // The then branch should be another if expression
  auto* innerIf = dynamic_cast<const NIfExpression*>(&outerIf->thenExpr);
  ASSERT_NE(innerIf, nullptr);

  // The outer else should be 3
  auto* elseExpr = dynamic_cast<const NInteger*>(&outerIf->elseExpr);
  ASSERT_NE(elseExpr, nullptr);
  EXPECT_EQ(elseExpr->value, 3);
}

TEST(ParserTest, IfExpressionInVariableDeclaration) {
  NBlock* block = parseOrFail("let x = if a then 1 else 0");
  ASSERT_NE(block, nullptr);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "x");

  auto* ifExpr = dynamic_cast<NIfExpression*>(varDecl->assignmentExpr);
  ASSERT_NE(ifExpr, nullptr);
}

TEST(ParserTest, IfExpressionInFunctionBody) {
  NBlock* block =
      parseOrFail("let max(a: int, b: int): int = if a > b then a else b");
  ASSERT_NE(block, nullptr);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "max");

  // Function body is a block with one expression statement
  ASSERT_EQ(funcDecl->block.statements.size(), 1);
  auto* bodyStmt =
      dynamic_cast<NExpressionStatement*>(funcDecl->block.statements[0]);
  ASSERT_NE(bodyStmt, nullptr);

  auto* ifExpr = dynamic_cast<const NIfExpression*>(&bodyStmt->expression);
  ASSERT_NE(ifExpr, nullptr);
}

// ============== Let Expression Tests ==============

TEST(ParserTest, SimpleLetExpression) {
  NBlock* block = parseOrFail("let x = 1 in x + 1");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  ASSERT_NE(exprStmt, nullptr);

  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 1);
  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  EXPECT_EQ(letExpr->bindings[0]->var->id.name, "x");
  EXPECT_EQ(letExpr->bindings[0]->var->type,
            nullptr); // type inferred, not yet set

  auto* initExpr =
      dynamic_cast<NInteger*>(letExpr->bindings[0]->var->assignmentExpr);
  ASSERT_NE(initExpr, nullptr);
  EXPECT_EQ(initExpr->value, 1);

  auto* bodyExpr = dynamic_cast<const NBinaryOperator*>(&letExpr->body);
  ASSERT_NE(bodyExpr, nullptr);
  EXPECT_EQ(bodyExpr->op, TPLUS);
}

TEST(ParserTest, LetExpressionMultipleBindings) {
  NBlock* block = parseOrFail("let x = 1 and y = 2 in x + y");
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->statements.size(), 1);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 2);

  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  EXPECT_EQ(letExpr->bindings[0]->var->id.name, "x");
  auto* initX =
      dynamic_cast<NInteger*>(letExpr->bindings[0]->var->assignmentExpr);
  ASSERT_NE(initX, nullptr);
  EXPECT_EQ(initX->value, 1);

  ASSERT_FALSE(letExpr->bindings[1]->isFunction);
  EXPECT_EQ(letExpr->bindings[1]->var->id.name, "y");
  auto* initY =
      dynamic_cast<NInteger*>(letExpr->bindings[1]->var->assignmentExpr);
  ASSERT_NE(initY, nullptr);
  EXPECT_EQ(initY->value, 2);
}

TEST(ParserTest, LetExpressionWithTypeAnnotation) {
  NBlock* block = parseOrFail("let x : int = 1 in x");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 1);
  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  EXPECT_EQ(letExpr->bindings[0]->var->id.name, "x");
  ASSERT_NE(letExpr->bindings[0]->var->type, nullptr);
  EXPECT_EQ(letExpr->bindings[0]->var->type->name, "int");
}

TEST(ParserTest, LetExpressionMixedTypeAnnotations) {
  NBlock* block =
      parseOrFail("let x : int = 1 and y = 2 and z : double = 3.0 in x");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 3);

  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  ASSERT_NE(letExpr->bindings[0]->var->type, nullptr);
  EXPECT_EQ(letExpr->bindings[0]->var->type->name, "int");
  ASSERT_FALSE(letExpr->bindings[1]->isFunction);
  EXPECT_EQ(letExpr->bindings[1]->var->type,
            nullptr); // type inferred, not yet set
  ASSERT_FALSE(letExpr->bindings[2]->isFunction);
  ASSERT_NE(letExpr->bindings[2]->var->type, nullptr);
  EXPECT_EQ(letExpr->bindings[2]->var->type->name, "double");
}

TEST(ParserTest, NestedLetExpression) {
  NBlock* block = parseOrFail("let x = 1 in let y = 2 in x + y");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* outerLet = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(outerLet, nullptr);
  ASSERT_FALSE(outerLet->bindings[0]->isFunction);
  EXPECT_EQ(outerLet->bindings[0]->var->id.name, "x");

  auto* innerLet = dynamic_cast<const NLetExpression*>(&outerLet->body);
  ASSERT_NE(innerLet, nullptr);
  ASSERT_FALSE(innerLet->bindings[0]->isFunction);
  EXPECT_EQ(innerLet->bindings[0]->var->id.name, "y");

  auto* bodyExpr = dynamic_cast<const NBinaryOperator*>(&innerLet->body);
  ASSERT_NE(bodyExpr, nullptr);
}

TEST(ParserTest, LetExpressionInVariableDeclaration) {
  NBlock* block = parseOrFail("let a = let x = 5 in x + 1");
  ASSERT_NE(block, nullptr);

  auto* varDecl = getFirstStatement<NVariableDeclaration>(block);
  ASSERT_NE(varDecl, nullptr);
  EXPECT_EQ(varDecl->id.name, "a");

  auto* letExpr = dynamic_cast<NLetExpression*>(varDecl->assignmentExpr);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  EXPECT_EQ(letExpr->bindings[0]->var->id.name, "x");
}

TEST(ParserTest, LetExpressionInFunctionBody) {
  NBlock* block = parseOrFail("let f(a: int): int = let b = 2 in a * b");
  ASSERT_NE(block, nullptr);

  auto* funcDecl = getFirstStatement<NFunctionDeclaration>(block);
  ASSERT_NE(funcDecl, nullptr);
  EXPECT_EQ(funcDecl->id.name, "f");

  ASSERT_EQ(funcDecl->block.statements.size(), 1);
  auto* bodyStmt =
      dynamic_cast<NExpressionStatement*>(funcDecl->block.statements[0]);
  ASSERT_NE(bodyStmt, nullptr);

  auto* letExpr = dynamic_cast<const NLetExpression*>(&bodyStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  EXPECT_EQ(letExpr->bindings[0]->var->id.name, "b");
}

TEST(ParserTest, LetExpressionWithIfBody) {
  NBlock* block = parseOrFail("let x = 1 in if x > 0 then x else 0");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);

  auto* ifExpr = dynamic_cast<const NIfExpression*>(&letExpr->body);
  ASSERT_NE(ifExpr, nullptr);
}

TEST(ParserTest, LetExpressionWithComplexInitializer) {
  NBlock* block = parseOrFail("let x = 1 + 2 * 3 in x");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);

  // Initializer should be 1 + (2 * 3) due to precedence
  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  auto* addOp =
      dynamic_cast<NBinaryOperator*>(letExpr->bindings[0]->var->assignmentExpr);
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(addOp->op, TPLUS);
}

TEST(ParserTest, LetExpressionThreeBindings) {
  NBlock* block = parseOrFail("let a = 1 and b = 2 and c = 3 in a + b + c");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 3);
  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  EXPECT_EQ(letExpr->bindings[0]->var->id.name, "a");
  ASSERT_FALSE(letExpr->bindings[1]->isFunction);
  EXPECT_EQ(letExpr->bindings[1]->var->id.name, "b");
  ASSERT_FALSE(letExpr->bindings[2]->isFunction);
  EXPECT_EQ(letExpr->bindings[2]->var->id.name, "c");
}

// ============== Function in Let Expression Tests ==============

TEST(ParserTest, LetExpressionWithFunction) {
  NBlock* block = parseOrFail("let f(x: int): int = x + 1 in f(5)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 1);

  // Check that the binding is a function
  ASSERT_TRUE(letExpr->bindings[0]->isFunction);
  auto* func = letExpr->bindings[0]->func;
  ASSERT_NE(func, nullptr);
  EXPECT_EQ(func->id.name, "f");
  ASSERT_NE(func->type, nullptr);
  EXPECT_EQ(func->type->name, "int");
  ASSERT_EQ(func->arguments.size(), 1);
  EXPECT_EQ(func->arguments[0]->id.name, "x");

  // Check that the body is a function call
  auto* call = dynamic_cast<const NMethodCall*>(&letExpr->body);
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->id.name, "f");
}

TEST(ParserTest, LetExpressionWithFunctionInferredReturnType) {
  NBlock* block = parseOrFail("let double(n: int) = n * 2 in double(5)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 1);

  // Check that the binding is a function with inferred return type
  ASSERT_TRUE(letExpr->bindings[0]->isFunction);
  auto* func = letExpr->bindings[0]->func;
  ASSERT_NE(func, nullptr);
  EXPECT_EQ(func->id.name, "double");
  EXPECT_EQ(func->type,
            nullptr); // Return type not yet inferred (set by type checker)
}

TEST(ParserTest, LetExpressionMultipleFunctions) {
  NBlock* block = parseOrFail(
      "let square(n: int): int = n * n and cube(n: int): int = n * n * n in "
      "square(2) + cube(2)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 2);

  // First binding: square
  ASSERT_TRUE(letExpr->bindings[0]->isFunction);
  EXPECT_EQ(letExpr->bindings[0]->func->id.name, "square");

  // Second binding: cube
  ASSERT_TRUE(letExpr->bindings[1]->isFunction);
  EXPECT_EQ(letExpr->bindings[1]->func->id.name, "cube");
}

TEST(ParserTest, LetExpressionMixedVariablesAndFunctions) {
  NBlock* block = parseOrFail("let x = 10 and f(y: int): int = y * 2 in f(x)");
  ASSERT_NE(block, nullptr);

  auto* exprStmt = getFirstStatement<NExpressionStatement>(block);
  auto* letExpr = dynamic_cast<const NLetExpression*>(&exprStmt->expression);
  ASSERT_NE(letExpr, nullptr);
  ASSERT_EQ(letExpr->bindings.size(), 2);

  // First binding: variable x
  ASSERT_FALSE(letExpr->bindings[0]->isFunction);
  EXPECT_EQ(letExpr->bindings[0]->var->id.name, "x");

  // Second binding: function f
  ASSERT_TRUE(letExpr->bindings[1]->isFunction);
  EXPECT_EQ(letExpr->bindings[1]->func->id.name, "f");
}
