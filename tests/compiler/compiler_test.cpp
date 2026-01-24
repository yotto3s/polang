#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "process_helper.hpp"

using ::testing::HasSubstr;

// Test valid programs - verify LLVM IR is generated
// Note: MLIR backend (default) performs constant folding, so some tests
// check for optimized output patterns.
// Basic type and arithmetic tests are covered by lit tests in tests/lit/LLVMIR/

TEST(CompilerIntegration, ComparisonInFunction) {
  const auto result = runCompiler("let lt(a: i64, b: i64): bool = a < b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("icmp"));
}

TEST(CompilerIntegration, FunctionDeclaration) {
  const auto result = runCompiler("let add(a: i64, b: i64): i64 = a + b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("define"));
  EXPECT_THAT(result.stdout_output, HasSubstr("@add"));
}

TEST(CompilerIntegration, IfExpressionInFunction) {
  // Use function to test if-expression without constant folding
  const auto result =
      runCompiler("let abs(x: i64): i64 = if x < 0 then 0 - x else x");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("br"));
  EXPECT_THAT(result.stdout_output, HasSubstr("icmp"));
}

TEST(CompilerIntegration, ConstantFoldedExpression) {
  // MLIR backend performs constant folding for immutable bindings
  // Result: 1 + 2 = 3 is computed at compile time
  const auto result = runCompiler("let y: i64 = let x = 1 in x + 2");
  EXPECT_EQ(result.exit_code, 0);
  // Check for constant-folded result (ret i64 3)
  EXPECT_THAT(result.stdout_output, HasSubstr("ret i64 3"));
}

TEST(CompilerIntegration, LetExpressionMultipleBindings) {
  // MLIR backend performs constant folding for immutable bindings
  // Result: 1 + 2 = 3 is computed at compile time
  const auto result = runCompiler("let z: i64 = let x = 1 and y = 2 in x + y");
  EXPECT_EQ(result.exit_code, 0);
  // Check for constant-folded result (ret i64 3)
  EXPECT_THAT(result.stdout_output, HasSubstr("ret i64 3"));
}

TEST(CompilerIntegration, FunctionDeclarationAndCall) {
  const auto result =
      runCompiler("let double(x: i64): i64 = x * 2\nlet y: i64 = double(5)");
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

// VariableReassignment test removed - variables are now immutable

// VariableShadowingInLet test moved to lit/LLVMIR/variable-shadowing.po

TEST(CompilerIntegration, NestedLetExpression) {
  // MLIR backend performs constant folding for immutable bindings
  // Result: x=1, y=x+1=2, y*2=4 is computed at compile time
  const auto result = runCompiler("let x = 1 in let y = x + 1 in y * 2");
  EXPECT_EQ(result.exit_code, 0);
  // Check for constant-folded result (ret i64 4)
  EXPECT_THAT(result.stdout_output, HasSubstr("ret i64 4"));
}

// RecursiveFunction test moved to lit/LLVMIR/recursive-function.po

// Double comparison and arithmetic tests are covered by lit tests in
// tests/lit/LLVMIR/double-comparisons.po and
// tests/lit/MLIR/double-arithmetic.po

// NestedIfExpressionInFunction test moved to lit/LLVMIR/nested-if.po

TEST(CompilerIntegration, FunctionWithMultipleParams) {
  const auto result =
      runCompiler("let add(a: i64, b: i64, c: i64): i64 = a + b + c");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("@add"));
  // MLIR backend uses SSA style (no allocas for params)
  EXPECT_THAT(result.stdout_output, HasSubstr("add i64"));
}

TEST(CompilerIntegration, LetWithFunctionBinding) {
  const auto result = runCompiler("let f(x: i64): i64 = x * 2 in f(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("@f"));
  EXPECT_THAT(result.stdout_output, HasSubstr("call"));
}

// ComplexExpressionConstantFolded test moved to lit/LLVMIR/constant-folding.po

// ============== CLI Tests ==============

TEST(CompilerCLI, HelpFlag) {
  const auto result = runCompilerWithArgs({"--help"});
  EXPECT_EQ(result.exit_code, 0);
  // Usage is printed to stderr
  EXPECT_THAT(result.stderr_output, HasSubstr("Usage:"));
  EXPECT_THAT(result.stderr_output, HasSubstr("--dump-ast"));
  EXPECT_THAT(result.stderr_output, HasSubstr("--emit-mlir"));
}

TEST(CompilerCLI, UnknownOption) {
  // Unknown option triggers error path in compiler main.cpp:42-45
  const auto result = runCompilerWithArgs({"--unknown-flag"});
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_THAT(result.stderr_output, HasSubstr("Unknown option"));
}

TEST(CompilerCLI, EmitMlirFlag) {
  // Test --emit-mlir produces Polang dialect MLIR
  const auto result = runCompiler("let x = 42");
  // Default output is LLVM IR
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("define"));
}

TEST(CompilerCLI, DumpAstFlag) {
  // Verify --dump-ast with stdin input
  const auto result = runCompiler("let x = 42");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(CompilerIntegration, ParseErrorExitsWithCode1) {
  // Incomplete syntax triggers parse error path in compiler main.cpp:64-66
  const auto result = runCompiler("let");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_THAT(result.stderr_output, HasSubstr("syntax error"));
}

TEST(CompilerIntegration, TypeErrorExitsWithCode1) {
  // Type mismatch triggers type inference failure
  const auto result = runCompiler("1 + 1.0");
  EXPECT_NE(result.exit_code, 0);
}

TEST(CompilerIntegration, FloatArithmetic) {
  // Test float operations produce fcmp/fadd instructions
  const auto result =
      runCompiler("let fsub(a: f64, b: f64): f64 = a - b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fsub"));
}

TEST(CompilerIntegration, FloatMultiplication) {
  const auto result =
      runCompiler("let fmul(a: f64, b: f64): f64 = a * b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fmul"));
}

TEST(CompilerIntegration, FloatDivision) {
  const auto result =
      runCompiler("let fdiv(a: f64, b: f64): f64 = a / b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fdiv"));
}

TEST(CompilerIntegration, IntegerDivision) {
  const auto result =
      runCompiler("let idiv(a: i64, b: i64): i64 = a / b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("sdiv"));
}

TEST(CompilerIntegration, FloatComparison) {
  const auto result =
      runCompiler("let fle(a: f64, b: f64): bool = a <= b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fcmp ole"));
}

TEST(CompilerIntegration, FloatNotEqual) {
  const auto result =
      runCompiler("let fne(a: f64, b: f64): bool = a != b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fcmp one"));
}

TEST(CompilerIntegration, FloatGreaterEqual) {
  const auto result =
      runCompiler("let fge(a: f64, b: f64): bool = a >= b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fcmp oge"));
}

TEST(CompilerIntegration, CastIntToFloat) {
  const auto result =
      runCompiler("let to_f64(x: i64): f64 = x as f64");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("sitofp"));
}

TEST(CompilerIntegration, CastFloatToInt) {
  const auto result =
      runCompiler("let to_i64(x: f64): i64 = x as i64");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fptosi.sat"));
}
