// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
#include <memory>
#include <string>

// clang-format off
// Parser headers
#include "parser/node.hpp"
#include "parser.hpp"
#include "parser/parser_api.hpp"
#include "parser/type_checker.hpp"
// clang-format on

// ============== checkIncremental Tests ==============
// These test that the TypeChecker can incrementally check new AST nodes
// while preserving its environment from previous evaluations.

class IncrementalTypeCheckerTest : public ::testing::Test {
protected:
  TypeChecker tc;

  // Parse source into an NBlock
  static std::unique_ptr<NBlock> parse(const std::string& source) {
    return polang_parse(source);
  }
};

TEST_F(IncrementalTypeCheckerTest, BasicVariableDeclaration) {
  // First evaluation: declare a variable
  auto ast1 = parse("x = 42");
  ASSERT_NE(ast1, nullptr);
  auto errors = tc.check(*ast1);
  EXPECT_TRUE(errors.empty());
}

TEST_F(IncrementalTypeCheckerTest, IncrementalCheckPreservesEnvironment) {
  // First evaluation: declare a variable
  auto ast1 = parse("x = 42");
  ASSERT_NE(ast1, nullptr);
  auto errors1 = tc.check(*ast1);
  EXPECT_TRUE(errors1.empty());

  // Second evaluation: use the variable — should succeed with checkIncremental
  auto ast2 = parse("x + 1");
  ASSERT_NE(ast2, nullptr);
  auto errors2 = tc.checkIncremental(*ast2);
  EXPECT_TRUE(errors2.empty());
  EXPECT_EQ(tc.getInferredType(), "i64");
}

TEST_F(IncrementalTypeCheckerTest, IncrementalCheckDetectsUndeclaredVariable) {
  // First evaluation: declare x
  auto ast1 = parse("x = 42");
  ASSERT_NE(ast1, nullptr);
  auto errors1 = tc.check(*ast1);
  EXPECT_TRUE(errors1.empty());

  // Second evaluation: use undeclared y — should fail
  auto ast2 = parse("y + 1");
  ASSERT_NE(ast2, nullptr);
  auto errors2 = tc.checkIncremental(*ast2);
  EXPECT_FALSE(errors2.empty());
}

TEST_F(IncrementalTypeCheckerTest, IncrementalFunctionDeclarationAndCall) {
  // First evaluation: declare a function
  auto ast1 = parse("f : i64 -> i64\nf(x) = x + 1");
  ASSERT_NE(ast1, nullptr);
  auto errors1 = tc.check(*ast1);
  EXPECT_TRUE(errors1.empty());

  // Second evaluation: call the function — should succeed
  auto ast2 = parse("f(10)");
  ASSERT_NE(ast2, nullptr);
  auto errors2 = tc.checkIncremental(*ast2);
  EXPECT_TRUE(errors2.empty());
  EXPECT_EQ(tc.getInferredType(), "i64");
}

TEST_F(IncrementalTypeCheckerTest, MultipleIncrementalEvaluations) {
  // Eval 1: declare variable
  auto ast1 = parse("a = 10");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Eval 2: declare another variable using first
  auto ast2 = parse("b = 20");
  ASSERT_NE(ast2, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast2).empty());

  // Eval 3: use both variables
  auto ast3 = parse("a + b");
  ASSERT_NE(ast3, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast3).empty());
  EXPECT_EQ(tc.getInferredType(), "i64");
}

TEST_F(IncrementalTypeCheckerTest, FunctionCallsAcrossEvaluations) {
  // Eval 1: declare add1
  auto ast1 = parse("add1 : i64 -> i64\nadd1(x) = x + 1");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Eval 2: declare add2 that calls add1
  auto ast2 = parse("add2 : i64 -> i64\nadd2(x) = add1(add1(x))");
  ASSERT_NE(ast2, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast2).empty());

  // Eval 3: call add2
  auto ast3 = parse("add2(10)");
  ASSERT_NE(ast3, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast3).empty());
  EXPECT_EQ(tc.getInferredType(), "i64");
}

TEST_F(IncrementalTypeCheckerTest, TypeMismatchDetected) {
  // Eval 1: declare integer variable
  auto ast1 = parse("x = 42");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Eval 2: try to use it in float expression — should fail
  auto ast2 = parse("x + 1.0");
  ASSERT_NE(ast2, nullptr);
  auto errors = tc.checkIncremental(*ast2);
  EXPECT_FALSE(errors.empty());
}

TEST_F(IncrementalTypeCheckerTest, PolymorphicFunctionAcrossEvaluations) {
  // Eval 1: declare polymorphic identity function
  auto ast1 = parse("identity(x) = x");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Eval 2: call with integer
  auto ast2 = parse("identity(42)");
  ASSERT_NE(ast2, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast2).empty());
  EXPECT_EQ(tc.getInferredType(), "i64");
}

TEST_F(IncrementalTypeCheckerTest, PolymorphicFunctionMultipleCallTypes) {
  // Eval 1: declare polymorphic identity function
  auto ast1 = parse("identity(x) = x");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Eval 2: call with integer
  auto ast2 = parse("identity(42)");
  ASSERT_NE(ast2, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast2).empty());
  EXPECT_EQ(tc.getInferredType(), "i64");

  // Eval 3: call with double — polymorphic, should also work
  auto ast3 = parse("identity(3.14)");
  ASSERT_NE(ast3, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast3).empty());
  EXPECT_EQ(tc.getInferredType(), "f64");
}

// ============== TypeCheckerSnapshot Tests ==============

TEST_F(IncrementalTypeCheckerTest, SnapshotSaveAndRestore) {
  // Eval 1: declare variable
  auto ast1 = parse("x = 42");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Save state
  auto snapshot = tc.saveState();

  // Eval 2: declare another variable incrementally
  auto ast2 = parse("y = 100");
  ASSERT_NE(ast2, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast2).empty());

  // Restore state — y should no longer be known
  tc.restoreState(snapshot);

  // Eval 3: try to use y — should fail since we restored
  auto ast3 = parse("y + 1");
  ASSERT_NE(ast3, nullptr);
  auto errors = tc.checkIncremental(*ast3);
  EXPECT_FALSE(errors.empty());

  // But x should still be known
  auto ast4 = parse("x + 1");
  ASSERT_NE(ast4, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast4).empty());
}

TEST_F(IncrementalTypeCheckerTest, SnapshotRollbackOnError) {
  // Eval 1: declare x
  auto ast1 = parse("x = 42");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Save state before risky eval
  auto snapshot = tc.saveState();

  // Eval 2: bad input
  auto ast2 = parse("undeclared_var + 1");
  ASSERT_NE(ast2, nullptr);
  auto errors = tc.checkIncremental(*ast2);
  EXPECT_FALSE(errors.empty());

  // Restore state
  tc.restoreState(snapshot);

  // Eval 3: x should still work
  auto ast3 = parse("x + 1");
  ASSERT_NE(ast3, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast3).empty());
  EXPECT_EQ(tc.getInferredType(), "i64");
}

TEST_F(IncrementalTypeCheckerTest, SnapshotPreservesFunctionSchemes) {
  // Eval 1: declare polymorphic function
  auto ast1 = parse("identity(x) = x");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Save state
  auto snapshot = tc.saveState();

  // Eval 2: declare something else
  auto ast2 = parse("y = 100");
  ASSERT_NE(ast2, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast2).empty());

  // Restore state (removes y, but keeps identity)
  tc.restoreState(snapshot);

  // Eval 3: identity should still work
  auto ast3 = parse("identity(42)");
  ASSERT_NE(ast3, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast3).empty());
  EXPECT_EQ(tc.getInferredType(), "i64");
}

// ============== Edge Cases ==============

TEST_F(IncrementalTypeCheckerTest, EmptyBlockIncremental) {
  // First check establishes environment
  auto ast1 = parse("x = 42");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Empty block should be fine
  NBlock emptyBlock;
  auto errors = tc.checkIncremental(emptyBlock);
  EXPECT_TRUE(errors.empty());

  // x should still be available
  auto ast3 = parse("x + 1");
  ASSERT_NE(ast3, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast3).empty());
}

TEST_F(IncrementalTypeCheckerTest, ClosureWithCrossEvalCapture) {
  // Eval 1: declare a variable
  auto ast1 = parse("base = 100");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Eval 2: declare a function that captures base
  auto ast2 = parse("add : i64 -> i64\nadd(x) = x + base");
  ASSERT_NE(ast2, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast2).empty());

  // Eval 3: call the function
  auto ast3 = parse("add(5)");
  ASSERT_NE(ast3, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast3).empty());
  EXPECT_EQ(tc.getInferredType(), "i64");
}

TEST_F(IncrementalTypeCheckerTest, DoubleDeclaration) {
  // Eval 1: declare x = 3.14
  auto ast1 = parse("x = 3.14");
  ASSERT_NE(ast1, nullptr);
  EXPECT_TRUE(tc.check(*ast1).empty());

  // Eval 2: use x in double arithmetic
  auto ast2 = parse("x + 1.0");
  ASSERT_NE(ast2, nullptr);
  EXPECT_TRUE(tc.checkIncremental(*ast2).empty());
  EXPECT_EQ(tc.getInferredType(), "f64");
}
