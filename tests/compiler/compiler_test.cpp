#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "process_helper.hpp"

using ::testing::HasSubstr;

// Test valid programs - verify LLVM IR is generated

TEST(CompilerIntegration, IntegerVariableDeclaration) {
  const auto result = runCompiler("let x: int = 42");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("i64 42"));
  EXPECT_THAT(result.stdout_output, HasSubstr("alloca"));
  EXPECT_THAT(result.stdout_output, HasSubstr("store"));
}

TEST(CompilerIntegration, DoubleVariableDeclaration) {
  const auto result = runCompiler("let x: double = 3.14");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("double"));
  EXPECT_THAT(result.stdout_output, HasSubstr("alloca"));
}

TEST(CompilerIntegration, BooleanTrue) {
  const auto result = runCompiler("let b: bool = true");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("i1 true"));
}

TEST(CompilerIntegration, BooleanFalse) {
  const auto result = runCompiler("let b: bool = false");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("i1 false"));
}

TEST(CompilerIntegration, BinaryAddition) {
  const auto result = runCompiler("let x: int = 1 + 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("add"));
}

TEST(CompilerIntegration, BinarySubtraction) {
  const auto result = runCompiler("let x: int = 5 - 3");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("sub"));
}

TEST(CompilerIntegration, BinaryMultiplication) {
  const auto result = runCompiler("let x: int = 4 * 3");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("mul"));
}

TEST(CompilerIntegration, BinaryDivision) {
  const auto result = runCompiler("let x: int = 10 / 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("sdiv"));
}

TEST(CompilerIntegration, Comparison) {
  const auto result = runCompiler("let b: bool = 1 < 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("icmp"));
}

TEST(CompilerIntegration, FunctionDeclaration) {
  const auto result = runCompiler("let add(a: int, b: int): int = a + b");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("define"));
  EXPECT_THAT(result.stdout_output, HasSubstr("@add"));
}

TEST(CompilerIntegration, IfExpression) {
  const auto result = runCompiler("let x: int = if true then 1 else 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("br"));
}

TEST(CompilerIntegration, LetExpression) {
  const auto result = runCompiler("let y: int = let x = 1 in x + 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("add"));
}

TEST(CompilerIntegration, LetExpressionMultipleBindings) {
  const auto result = runCompiler("let z: int = let x = 1 and y = 2 in x + y");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("add"));
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
  const auto result = runCompiler("let x = 5\nx = 10\nx");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("store"));
}

TEST(CompilerIntegration, VariableShadowingInLet) {
  const auto result = runCompiler("let x = 1 in let x = 2 in x");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("alloca"));
}

TEST(CompilerIntegration, NestedLetExpression) {
  const auto result = runCompiler("let x = 1 in let y = x + 1 in y * 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("mul"));
}

TEST(CompilerIntegration, RecursiveFunction) {
  const auto result = runCompiler("let factorial(n: int): int = if n <= 1 then "
                                  "1 else n * factorial(n - 1)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("@factorial"));
  EXPECT_THAT(result.stdout_output, HasSubstr("call"));
}

TEST(CompilerIntegration, DoubleComparison) {
  const auto result = runCompiler("let b: bool = 1.5 < 2.5");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fcmp"));
}

TEST(CompilerIntegration, DoubleArithmetic) {
  const auto result = runCompiler("1.5 + 2.5");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("fadd"));
}

TEST(CompilerIntegration, NestedIfExpression) {
  const auto result = runCompiler("if true then if false then 1 else 2 else 3");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("then:"));
  EXPECT_THAT(result.stdout_output, HasSubstr("else:"));
}

TEST(CompilerIntegration, FunctionWithMultipleParams) {
  const auto result =
      runCompiler("let add(a: int, b: int, c: int): int = a + b + c");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("@add"));
  // Parameters are numbered in LLVM IR, variables stored as allocas
  EXPECT_THAT(result.stdout_output, HasSubstr("%a = alloca"));
  EXPECT_THAT(result.stdout_output, HasSubstr("%b = alloca"));
  EXPECT_THAT(result.stdout_output, HasSubstr("%c = alloca"));
}

TEST(CompilerIntegration, LetWithFunctionBinding) {
  const auto result = runCompiler("let f(x: int): int = x * 2 in f(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("@f"));
  EXPECT_THAT(result.stdout_output, HasSubstr("call"));
}

TEST(CompilerIntegration, ComplexExpression) {
  const auto result = runCompiler("(1 + 2) * (3 + 4)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("add"));
  EXPECT_THAT(result.stdout_output, HasSubstr("mul"));
}
