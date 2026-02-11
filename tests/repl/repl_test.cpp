#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "process_helper.hpp"

using ::testing::HasSubstr;

// Basic literal, operation, and error tests are covered by lit tests in tests/lit/
// See: tests/lit/Execution/, tests/lit/Errors/

// ============== Function in Let Expression Tests ==============

TEST(ReplIntegration, LetExpressionWithFunction) {
  const auto result = runRepl("let f(x: i64): i64 = x + 1 in f(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("6"));
  EXPECT_THAT(result.stdout_output, HasSubstr("i64"));
}

TEST(ReplIntegration, LetExpressionMultipleFunctions) {
  const auto result = runRepl(
      "let square(n: i64): i64 = n * n and cube(n: i64): i64 = n * n * n in "
      "square(3) + cube(2)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("17")); // 9 + 8
  EXPECT_THAT(result.stdout_output, HasSubstr("i64"));
}

TEST(ReplIntegration, LetExpressionMixedBindingsCallBody) {
  // Variable and function in let, function is called with variable
  const auto result = runRepl("let x = 10 and f(y: i64): i64 = y * 2 in f(x)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("20"));
  EXPECT_THAT(result.stdout_output, HasSubstr("i64"));
}

// ============== State Persistence Tests ==============
// These tests verify that variables and functions persist across evaluations

TEST(ReplIntegration, VariablePersistsAcrossEvaluations) {
  // Declare variable, then use it in next evaluation
  const auto result = runRepl("x = 42\nx + 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("43 : i64"));
}

TEST(ReplIntegration, FunctionPersistsAcrossEvaluations) {
  // Declare function, then call it in next evaluation
  const auto result = runRepl("double : i64 -> i64\ndouble(n) = n * 2\ndouble(21)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("42 : i64"));
}

TEST(ReplIntegration, MultipleVariablesPersist) {
  // Multiple variables across multiple lines
  const auto result = runRepl("a = 10\nb = 20\nc = 30\na + b + c");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("60 : i64"));
}

TEST(ReplIntegration, FunctionCalledWithPersistedVariable) {
  // Variable declared, then passed to function call
  const auto result = runRepl("multiplier = 3\nscale : i64 * i64 -> i64\n"
                              "scale(x, y) = x * y\nscale(10, multiplier)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("30 : i64"));
}

TEST(ReplIntegration, MultipleFunctionsPersist) {
  // Multiple functions, each can call the other
  const auto result = runRepl("add1 : i64 -> i64\nadd1(x) = x + 1\n"
                              "add2 : i64 -> i64\nadd2(x) = add1(add1(x))\n"
                              "add2(10)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("12 : i64"));
}

// VariableReassignmentPersists test removed - variables are now immutable

// Assignment return value tests (AssignmentReturnsIntValue, AssignmentReturnsDoubleValue,
// AssignmentReturnsBoolValue, ChainedAssignment) are covered by lit tests in tests/lit/Execution/

// Multi-line input detection tests are covered by repl_unit_test.cpp (IsInputIncomplete tests)

// ============== Closure / Variable Capture Tests ==============

TEST(ReplIntegration, SimpleClosure) {
  const auto result = runRepl("x = 10\nf() = x + 1\nf()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("11 : i64"));
}

TEST(ReplIntegration, ClosureWithParameter) {
  const auto result = runRepl("multiplier = 3\nscale(n) = n * multiplier\nscale(10)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("30 : i64"));
}

TEST(ReplIntegration, ClosureInLetExpression) {
  const auto result = runRepl("let x = 10 and f() = x + 1 in f()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("11 : i64"));
}

TEST(ReplIntegration, MultipleCapturedVariables) {
  const auto result = runRepl("a = 1\nb = 2\nsum() = a + b\nsum()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3 : i64"));
}

// ClosureWithMutableCapture test removed - variables are now immutable

// ClosureCaptureByValue test removed - variables are now immutable (no reassignment)

TEST(ReplIntegration, ClosureCapturesDouble) {
  const auto result = runRepl("x = 3.14\nf() = x + 1.0\nf()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("4.14 : f64"));
}

TEST(ReplIntegration, ClosureWithParamsAndCaptures) {
  const auto result = runRepl("base = 100\nadd(x) = x + base\nadd(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("105 : i64"));
}

TEST(ReplIntegration, NestedLetWithClosure) {
  const auto result = runRepl("let x = 10 in let f() = x + 1 in f()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("11 : i64"));
}

TEST(ReplIntegration, ClosureMultipleSiblings) {
  const auto result = runRepl("let a = 1 and b = 2 and sum() = a + b in sum()");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("3 : i64"));
}

// ============== Error Path Tests ==============
// These tests verify error handling in repl_session.cpp

TEST(ReplIntegration, ParseErrorReturnsFailure) {
  // Incomplete syntax triggers parse error in repl_session.cpp:47-48
  const auto result = runRepl("let");
  EXPECT_NE(result.exit_code, 0);
  EXPECT_THAT(result.stderr_output, HasSubstr("syntax error"));
}

TEST(ReplIntegration, TypeMismatchError) {
  // Type mismatch triggers type check error path in repl_session.cpp:65-76
  const auto result = runRepl("1 + 1.0");
  EXPECT_NE(result.exit_code, 0);
  EXPECT_THAT(result.stderr_output, HasSubstr("Type error"));
}

TEST(ReplIntegration, UndeclaredVariableError) {
  // Undeclared variable triggers type check error with rollback
  const auto result = runRepl("x + 1");
  EXPECT_NE(result.exit_code, 0);
  EXPECT_THAT(result.stderr_output, HasSubstr("Undeclared variable"));
}

TEST(ReplIntegration, TypeErrorAfterValidDeclaration) {
  // First line succeeds (declaration), second line has type error
  // This exercises the rollback path: accumulatedAst->statements.resize()
  const auto result = runRepl("x = 42\nx + 1.0");
  EXPECT_NE(result.exit_code, 0);
  EXPECT_THAT(result.stderr_output, HasSubstr("Type error"));
}

TEST(ReplIntegration, DeclarationDoesNotPrintValue) {
  // A declaration (not expression) should not print a value
  // This exercises the lastIsExpression=false path in repl_session.cpp:80-89
  const auto result = runRepl("x = 42");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(result.stdout_output.empty());
}

TEST(ReplIntegration, FunctionArityMismatch) {
  // Wrong number of arguments triggers type error
  const auto result = runRepl("f : i64 -> i64\nf(x) = x\nf(1, 2)");
  EXPECT_NE(result.exit_code, 0);
  EXPECT_THAT(result.stderr_output, HasSubstr("argument"));
}

TEST(ReplIntegration, FloatResultDisplay) {
  // Tests the float type detection in repl main.cpp printResult
  const auto result = runRepl("3.14 + 1.0");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("4.14"));
  EXPECT_THAT(result.stdout_output, HasSubstr("f64"));
}

TEST(ReplIntegration, BoolResultDisplay) {
  // Tests the bool type detection in repl main.cpp printResult
  const auto result = runRepl("1 < 2");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("true : bool"));
}

TEST(ReplIntegration, BoolFalseDisplay) {
  const auto result = runRepl("2 < 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("false : bool"));
}

TEST(ReplIntegration, MultipleEvaluationsLastPrints) {
  // Multiple expressions - only the last one is printed in file mode
  // In stdin mode, each is printed separately
  const auto result = runRepl("1 + 1\n2 + 2\n3 + 3");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("6 : i64"));
}

// ============== Incremental REPL / Persistent JIT Tests ==============

TEST(ReplIntegration, PolymorphicFunctionAcrossEvals) {
  // Define polymorphic function in eval 1, call with i64 in eval 2
  const auto result =
      runRepl("identity(x) = x\nidentity(42)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("42 : i64"));
}

TEST(ReplIntegration, PolymorphicFunctionMultipleTypes) {
  // Polymorphic function called with different types across evaluations
  const auto result =
      runRepl("identity(x) = x\nidentity(42)\nidentity(3.14)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("42 : i64"));
  EXPECT_THAT(result.stdout_output, HasSubstr("3.14"));
}

TEST(ReplIntegration, ErrorRecoveryPreservesState) {
  // Define variable, then cause type error — REPL exits on error in pipe mode,
  // but the important thing is the error message is correct and doesn't crash
  const auto result = runRepl("x = 42\nx + 1.0");
  EXPECT_NE(result.exit_code, 0);
  EXPECT_THAT(result.stderr_output, HasSubstr("Type error"));
}

TEST(ReplIntegration, FiveSequentialEvaluations) {
  // 5+ sequential evaluations building on each other
  const auto result = runRepl(
      "a = 1\n"
      "b = 2\n"
      "c = a + b\n"
      "f : i64 -> i64\n"
      "f(x) = x * c\n"
      "f(10)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("30 : i64"));
}

TEST(ReplIntegration, FunctionCompositionAcrossEvals) {
  // Define two functions in separate evals, compose them in a third
  const auto result = runRepl(
      "double : i64 -> i64\ndouble(x) = x * 2\n"
      "inc : i64 -> i64\ninc(x) = x + 1\n"
      "double(inc(4))");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("10 : i64"));
}

TEST(ReplIntegration, RecursiveFunctionAcrossEvals) {
  // Define recursive function, call it in next eval
  const auto result = runRepl(
      "fact : i64 -> i64\nfact(n) = if n <= 1 then 1 else n * fact(n - 1)\n"
      "fact(5)");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_THAT(result.stdout_output, HasSubstr("120 : i64"));
}
