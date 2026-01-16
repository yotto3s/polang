#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "process_helper.hpp"

using ::testing::HasSubstr;

// Test valid programs - verify LLVM IR is generated
// Note: MLIR backend (default) performs constant folding, so some tests
// check for optimized output patterns.

TEST(CompilerIntegration, IntegerVariableDeclaration) {
  const auto result = runCompiler("let x: int = 42");
  EXPECT_EQ(result.exit_code, 0);
  // MLIR backend constant-folds to direct return
  EXPECT_THAT(result.stdout_output, HasSubstr("i64"));
  EXPECT_THAT(result.stdout_output, HasSubstr("42"));
}

TEST(CompilerIntegration, DoubleVariableDeclaration) {
  // Use a function to test double type since main() always returns i64
  // Just define the function without calling it at top level
  const auto result = runCompiler("let getDouble(x: double): double = x");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("double"));
}

TEST(CompilerIntegration, BooleanInFunction) {
  // Use a function to test booleans since main returns int
  const auto result = runCompiler(
      "let isTrue(b: bool): int = if b then 1 else 0\nisTrue(true)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("i1"));
}

TEST(CompilerIntegration, BinaryAdditionInFunction) {
  // Use function to avoid constant folding
  const auto result = runCompiler("let add(a: int, b: int): int = a + b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("add"));
}

TEST(CompilerIntegration, BinarySubtractionInFunction) {
  const auto result = runCompiler("let sub(a: int, b: int): int = a - b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("sub"));
}

TEST(CompilerIntegration, BinaryMultiplicationInFunction) {
  const auto result = runCompiler("let mul(a: int, b: int): int = a * b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("mul"));
}

TEST(CompilerIntegration, BinaryDivisionInFunction) {
  const auto result = runCompiler("let div(a: int, b: int): int = a / b");
  EXPECT_EQ(result.exit_code, 0);
  // MLIR uses sdiv for signed division
  EXPECT_THAT(result.stdout_output, HasSubstr("sdiv"));
}

TEST(CompilerIntegration, ComparisonInFunction) {
  const auto result = runCompiler("let lt(a: int, b: int): bool = a < b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("icmp"));
}

TEST(CompilerIntegration, FunctionDeclaration) {
  const auto result = runCompiler("let add(a: int, b: int): int = a + b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("define"));
  EXPECT_THAT(result.stdout_output, HasSubstr("@add"));
}

TEST(CompilerIntegration, IfExpressionInFunction) {
  // Use function to test if-expression without constant folding
  const auto result =
      runCompiler("let abs(x: int): int = if x < 0 then 0 - x else x");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("br"));
  EXPECT_THAT(result.stdout_output, HasSubstr("icmp"));
}

TEST(CompilerIntegration, ConstantFoldedExpression) {
  // Note: MLIR backend uses allocas for let bindings, so no constant folding
  const auto result = runCompiler("let y: int = let x = 1 in x + 2");
  EXPECT_EQ(result.exit_code, 0);
  // Check for alloca and add operations
  EXPECT_THAT(result.stdout_output, HasSubstr("alloca"));
  EXPECT_THAT(result.stdout_output, HasSubstr("add i64"));
}

TEST(CompilerIntegration, LetExpressionMultipleBindings) {
  // Note: MLIR backend uses allocas for let bindings, so no constant folding
  const auto result = runCompiler("let z: int = let x = 1 and y = 2 in x + y");
  EXPECT_EQ(result.exit_code, 0);
  // Check for multiple allocas and add operation
  EXPECT_THAT(result.stdout_output, HasSubstr("alloca"));
  EXPECT_THAT(result.stdout_output, HasSubstr("add i64"));
}

TEST(CompilerIntegration, FunctionDeclarationAndCall) {
  const auto result =
      runCompiler("let double(x: int): int = x * 2\nlet y: int = double(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("call"));
}

TEST(CompilerIntegration, ModuleDefinition) {
  const auto result = runCompiler("let x = 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("ModuleID"));
  EXPECT_THAT(result.stdout_output, HasSubstr("define"));
  EXPECT_THAT(result.stdout_output, HasSubstr("@main"));
}

// Test error handling - verify exit code 1 and error message

TEST(CompilerIntegration, SyntaxError) {
  const auto result = runCompiler("let x =");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_THAT(result.stderr_output, HasSubstr("error"));
}

TEST(CompilerIntegration, SyntaxErrorMissingParen) {
  const auto result = runCompiler("let foo (: int = 1");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_THAT(result.stderr_output, HasSubstr("error"));
}

TEST(CompilerIntegration, TypeError) {
  const auto result = runCompiler("let x: int = 1 + 1.0");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_THAT(result.stderr_output, HasSubstr("error"));
}

TEST(CompilerIntegration, TypeErrorInIfCondition) {
  const auto result = runCompiler("let x: int = if 1 then 2 else 3");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_THAT(result.stderr_output, HasSubstr("error"));
}

// ============== Additional CodeGen Tests ==============

TEST(CompilerIntegration, VariableReassignment) {
  const auto result = runCompiler("let mut x = 5\nx <- 10\nx");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("store"));
}

TEST(CompilerIntegration, VariableShadowingInLet) {
  // Note: MLIR backend uses allocas for let bindings, so no constant folding
  // The inner x shadows the outer x
  const auto result = runCompiler("let x = 1 in let x = 2 in x");
  EXPECT_EQ(result.exit_code, 0);
  // Check for allocas and load operation
  EXPECT_THAT(result.stdout_output, HasSubstr("alloca"));
  EXPECT_THAT(result.stdout_output, HasSubstr("load"));
}

TEST(CompilerIntegration, NestedLetExpression) {
  // Note: MLIR backend uses allocas for let bindings, so no constant folding
  const auto result = runCompiler("let x = 1 in let y = x + 1 in y * 2");
  EXPECT_EQ(result.exit_code, 0);
  // Check for allocas and arithmetic operations
  EXPECT_THAT(result.stdout_output, HasSubstr("alloca"));
  EXPECT_THAT(result.stdout_output, HasSubstr("add i64"));
  EXPECT_THAT(result.stdout_output, HasSubstr("mul i64"));
}

TEST(CompilerIntegration, RecursiveFunction) {
  const auto result = runCompiler("let factorial(n: int): int = if n <= 1 then "
                                  "1 else n * factorial(n - 1)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("@factorial"));
  EXPECT_THAT(result.stdout_output, HasSubstr("call"));
}

TEST(CompilerIntegration, DoubleComparisonInFunction) {
  // Use function to get actual fcmp instruction
  const auto result =
      runCompiler("let cmp(a: double, b: double): bool = a < b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fcmp"));
}

TEST(CompilerIntegration, DoubleArithmeticInFunction) {
  // Use function to get actual fadd instruction
  const auto result =
      runCompiler("let add(a: double, b: double): double = a + b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fadd"));
}

TEST(CompilerIntegration, NestedIfExpressionInFunction) {
  // Use function to avoid constant folding
  const auto result = runCompiler(
      "let f(x: int): int = if x > 0 then if x > 10 then 1 else 2 else 3");
  EXPECT_EQ(result.exit_code, 0);
  // MLIR uses numeric labels, just check for branches
  EXPECT_THAT(result.stdout_output, HasSubstr("br"));
  EXPECT_THAT(result.stdout_output, HasSubstr("icmp"));
}

TEST(CompilerIntegration, FunctionWithMultipleParams) {
  const auto result =
      runCompiler("let add(a: int, b: int, c: int): int = a + b + c");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("@add"));
  // MLIR backend uses SSA style (no allocas for params)
  EXPECT_THAT(result.stdout_output, HasSubstr("add i64"));
}

TEST(CompilerIntegration, LetWithFunctionBinding) {
  const auto result = runCompiler("let f(x: int): int = x * 2 in f(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("@f"));
  EXPECT_THAT(result.stdout_output, HasSubstr("call"));
}

TEST(CompilerIntegration, ComplexExpressionConstantFolded) {
  // MLIR backend constant-folds (1 + 2) * (3 + 4) = 3 * 7 = 21
  const auto result = runCompiler("(1 + 2) * (3 + 4)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("ret i64 21"));
}
