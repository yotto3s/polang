#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "process_helper.hpp"

using ::testing::HasSubstr;

// Test valid programs - verify LLVM IR is generated
// Note: MLIR backend (default) performs constant folding, so some tests
// check for optimized output patterns.
// Basic type and arithmetic tests are covered by lit tests in tests/lit/LLVMIR/

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
  EXPECT_THAT(result.stdout_output, HasSubstr("@__polang_entry"));
}

// Test error handling - verify exit code 1 and error message
// Basic syntax error test is covered by lit tests in tests/lit/Errors/

// SyntaxErrorMissingParen test moved to lit/Errors/syntax-error-paren.po

// TypeError test already covered by lit/Errors/type-mismatch.po

// TypeErrorInIfCondition test moved to lit/Errors/type-error-if-condition.po

// ============== Additional CodeGen Tests ==============

TEST(CompilerIntegration, VariableReassignment) {
  const auto result = runCompiler("let mut x = 5\nx <- 10\nx");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("store"));
}

// VariableShadowingInLet test moved to lit/LLVMIR/variable-shadowing.po

TEST(CompilerIntegration, NestedLetExpression) {
  // Note: MLIR backend uses allocas for let bindings, so no constant folding
  const auto result = runCompiler("let x = 1 in let y = x + 1 in y * 2");
  EXPECT_EQ(result.exit_code, 0);
  // Check for allocas and arithmetic operations
  EXPECT_THAT(result.stdout_output, HasSubstr("alloca"));
  EXPECT_THAT(result.stdout_output, HasSubstr("add i64"));
  EXPECT_THAT(result.stdout_output, HasSubstr("mul i64"));
}

// RecursiveFunction test moved to lit/LLVMIR/recursive-function.po

// Double comparison and arithmetic tests are covered by lit tests in
// tests/lit/LLVMIR/double-comparisons.po and tests/lit/MLIR/double-arithmetic.po

// NestedIfExpressionInFunction test moved to lit/LLVMIR/nested-if.po

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

// ComplexExpressionConstantFolded test moved to lit/LLVMIR/constant-folding.po
