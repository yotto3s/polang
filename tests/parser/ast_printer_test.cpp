#include "parser/ast_printer.hpp"
#include "parser_test_helper.hpp"
#include <sstream>

// ============== Literal Tests ==============

TEST(ASTPrinterTest, PrintInteger) {
  NBlock* block = parseOrFail("42");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NBlock"), std::string::npos);
  EXPECT_NE(result.find("NExpressionStatement"), std::string::npos);
  EXPECT_NE(result.find("NInteger 42"), std::string::npos);
}

TEST(ASTPrinterTest, PrintDouble) {
  NBlock* block = parseOrFail("3.14");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  EXPECT_NE(out.str().find("NDouble 3.14"), std::string::npos);
}

TEST(ASTPrinterTest, PrintBooleanTrue) {
  NBlock* block = parseOrFail("true");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  EXPECT_NE(out.str().find("NBoolean true"), std::string::npos);
}

TEST(ASTPrinterTest, PrintBooleanFalse) {
  NBlock* block = parseOrFail("false");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  EXPECT_NE(out.str().find("NBoolean false"), std::string::npos);
}

TEST(ASTPrinterTest, PrintIdentifier) {
  NBlock* block = parseOrFail("myVar");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  EXPECT_NE(out.str().find("NIdentifier 'myVar'"), std::string::npos);
}

// ============== Binary Operator Tests ==============

TEST(ASTPrinterTest, PrintBinaryOperatorAdd) {
  NBlock* block = parseOrFail("1 + 2");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NBinaryOperator '+'"), std::string::npos);
  EXPECT_NE(result.find("NInteger 1"), std::string::npos);
  EXPECT_NE(result.find("NInteger 2"), std::string::npos);
}

TEST(ASTPrinterTest, PrintBinaryOperatorCompare) {
  NBlock* block = parseOrFail("x > 5");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  EXPECT_NE(out.str().find("NBinaryOperator '>'"), std::string::npos);
}

// ============== Declaration Tests ==============

TEST(ASTPrinterTest, PrintVariableDeclaration) {
  NBlock* block = parseOrFail("let x = 5");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NVariableDeclaration 'x'"), std::string::npos);
  EXPECT_NE(result.find("NInteger 5"), std::string::npos);
}

TEST(ASTPrinterTest, PrintMutableVariable) {
  NBlock* block = parseOrFail("let mut y = 10");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  EXPECT_NE(out.str().find("NVariableDeclaration 'y' mut"), std::string::npos);
}

TEST(ASTPrinterTest, PrintTypedVariable) {
  NBlock* block = parseOrFail("let z: int = 42");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  EXPECT_NE(out.str().find("NVariableDeclaration 'z' : int"),
            std::string::npos);
}

TEST(ASTPrinterTest, PrintFunctionDeclaration) {
  NBlock* block = parseOrFail("let add(a: int, b: int): int = a + b");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NFunctionDeclaration 'add'"), std::string::npos);
  EXPECT_NE(result.find("a: int"), std::string::npos);
  EXPECT_NE(result.find("b: int"), std::string::npos);
  EXPECT_NE(result.find("-> int"), std::string::npos);
}

// ============== Control Flow Tests ==============

TEST(ASTPrinterTest, PrintIfExpression) {
  NBlock* block = parseOrFail("if x > 0 then 1 else 2");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NIfExpression"), std::string::npos);
  EXPECT_NE(result.find("condition:"), std::string::npos);
  EXPECT_NE(result.find("then:"), std::string::npos);
  EXPECT_NE(result.find("else:"), std::string::npos);
}

TEST(ASTPrinterTest, PrintLetExpression) {
  NBlock* block = parseOrFail("let x = 1 and y = 2 in x + y");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NLetExpression"), std::string::npos);
  EXPECT_NE(result.find("body:"), std::string::npos);
}

// ============== Method Call Tests ==============

TEST(ASTPrinterTest, PrintMethodCall) {
  NBlock* block = parseOrFail("foo(1, 2)");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NMethodCall 'foo'"), std::string::npos);
  EXPECT_NE(result.find("NInteger 1"), std::string::npos);
  EXPECT_NE(result.find("NInteger 2"), std::string::npos);
}

// ============== Tree Structure Tests ==============

TEST(ASTPrinterTest, TreeConnectors) {
  NBlock* block = parseOrFail("1 + 2 * 3");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  // Should have proper tree connectors
  EXPECT_NE(result.find("|-"), std::string::npos);
  EXPECT_NE(result.find("`-"), std::string::npos);
}

TEST(ASTPrinterTest, MultipleStatements) {
  NBlock* block = parseOrFail("let x = 1\nlet y = 2\nx + y");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NVariableDeclaration 'x'"), std::string::npos);
  EXPECT_NE(result.find("NVariableDeclaration 'y'"), std::string::npos);
  EXPECT_NE(result.find("NBinaryOperator '+'"), std::string::npos);
}

// ============== Assignment Tests ==============

TEST(ASTPrinterTest, PrintAssignment) {
  NBlock* block = parseOrFail("x <- 10");
  std::ostringstream out;
  ASTPrinter printer(out);
  printer.print(*block);

  const std::string result = out.str();
  EXPECT_NE(result.find("NAssignment"), std::string::npos);
  EXPECT_NE(result.find("NIdentifier 'x'"), std::string::npos);
  EXPECT_NE(result.find("NInteger 10"), std::string::npos);
}
