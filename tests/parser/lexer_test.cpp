// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
#include <string>
#include <vector>

// Parser headers (includes LLVM via node.hpp)
// clang-format off
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison types
// clang-format on

// Token type shorthand
using token = yy::parser::token;

// Flex buffer type and functions
typedef struct yy_buffer_state* YY_BUFFER_STATE;
extern YY_BUFFER_STATE yy_scan_string(const char* str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
extern yy::parser::symbol_type yylex();

// Helper to get all tokens from a source string
std::vector<int> tokenize(const std::string& source) {
  std::vector<int> tokens;
  YY_BUFFER_STATE buffer = yy_scan_string(source.c_str());
  while (true) {
    auto sym = yylex();
    if (sym.kind() == yy::parser::symbol_kind::S_YYEOF) {
      break;
    }
    tokens.push_back(static_cast<int>(sym.kind()));
  }
  yy_delete_buffer(buffer);
  return tokens;
}

// Helper to get token with its string value
struct TokenInfo {
  int token;
  std::string value;
};

std::vector<TokenInfo> tokenizeWithValues(const std::string& source) {
  std::vector<TokenInfo> tokens;
  YY_BUFFER_STATE buffer = yy_scan_string(source.c_str());
  while (true) {
    auto sym = yylex();
    auto kind = sym.kind();
    if (kind == yy::parser::symbol_kind::S_YYEOF) {
      break;
    }
    TokenInfo info;
    info.token = static_cast<int>(kind);
    // For tokens that store string values, extract from the variant
    if (kind == yy::parser::symbol_kind::S_TIDENTIFIER ||
        kind == yy::parser::symbol_kind::S_TINTEGER ||
        kind == yy::parser::symbol_kind::S_TDOUBLE) {
      info.value = sym.value.as<std::string>();
    }
    tokens.push_back(info);
  }
  yy_delete_buffer(buffer);
  return tokens;
}

// Symbol kind constants for easier comparison
constexpr auto S_TLET = yy::parser::symbol_kind::S_TLET;
constexpr auto S_TFUN = yy::parser::symbol_kind::S_TFUN;
constexpr auto S_TIN = yy::parser::symbol_kind::S_TIN;
constexpr auto S_TIF = yy::parser::symbol_kind::S_TIF;
constexpr auto S_TTHEN = yy::parser::symbol_kind::S_TTHEN;
constexpr auto S_TELSE = yy::parser::symbol_kind::S_TELSE;
constexpr auto S_TIDENTIFIER = yy::parser::symbol_kind::S_TIDENTIFIER;
constexpr auto S_TINTEGER = yy::parser::symbol_kind::S_TINTEGER;
constexpr auto S_TDOUBLE = yy::parser::symbol_kind::S_TDOUBLE;
constexpr auto S_TPLUS = yy::parser::symbol_kind::S_TPLUS;
constexpr auto S_TMINUS = yy::parser::symbol_kind::S_TMINUS;
constexpr auto S_TMUL = yy::parser::symbol_kind::S_TMUL;
constexpr auto S_TDIV = yy::parser::symbol_kind::S_TDIV;
constexpr auto S_TEQUAL = yy::parser::symbol_kind::S_TEQUAL;
constexpr auto S_TCEQ = yy::parser::symbol_kind::S_TCEQ;
constexpr auto S_TCNE = yy::parser::symbol_kind::S_TCNE;
constexpr auto S_TCLT = yy::parser::symbol_kind::S_TCLT;
constexpr auto S_TCLE = yy::parser::symbol_kind::S_TCLE;
constexpr auto S_TCGT = yy::parser::symbol_kind::S_TCGT;
constexpr auto S_TCGE = yy::parser::symbol_kind::S_TCGE;
constexpr auto S_TARROW = yy::parser::symbol_kind::S_TARROW;
constexpr auto S_TLPAREN = yy::parser::symbol_kind::S_TLPAREN;
constexpr auto S_TRPAREN = yy::parser::symbol_kind::S_TRPAREN;
constexpr auto S_TLBRACE = yy::parser::symbol_kind::S_TLBRACE;
constexpr auto S_TRBRACE = yy::parser::symbol_kind::S_TRBRACE;
constexpr auto S_TCOMMA = yy::parser::symbol_kind::S_TCOMMA;
constexpr auto S_TCOLON = yy::parser::symbol_kind::S_TCOLON;
constexpr auto S_TDOT = yy::parser::symbol_kind::S_TDOT;

// ============== Keyword Tests ==============

TEST(LexerTest, KeywordLet) {
  auto tokens = tokenize("let");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLET));
}

TEST(LexerTest, KeywordFun) {
  auto tokens = tokenize("fun");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TFUN));
}

TEST(LexerTest, KeywordIn) {
  auto tokens = tokenize("in");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TIN));
}

TEST(LexerTest, KeywordIf) {
  auto tokens = tokenize("if");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TIF));
}

TEST(LexerTest, KeywordThen) {
  auto tokens = tokenize("then");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TTHEN));
}

TEST(LexerTest, KeywordElse) {
  auto tokens = tokenize("else");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TELSE));
}

// ============== Identifier Tests ==============

TEST(LexerTest, SimpleIdentifier) {
  auto tokens = tokenizeWithValues("foo");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[0].value, "foo");
}

TEST(LexerTest, IdentifierWithUnderscore) {
  auto tokens = tokenizeWithValues("my_var");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[0].value, "my_var");
}

TEST(LexerTest, IdentifierStartingWithUnderscore) {
  auto tokens = tokenizeWithValues("_private");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[0].value, "_private");
}

TEST(LexerTest, IdentifierWithNumbers) {
  auto tokens = tokenizeWithValues("var123");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[0].value, "var123");
}

TEST(LexerTest, KeywordPrefixIdentifier) {
  // "letter" starts with "let" but should be identifier
  auto tokens = tokenizeWithValues("letter");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[0].value, "letter");
}

TEST(LexerTest, IfPrefixIdentifier) {
  // "iffy" starts with "if" but should be identifier
  auto tokens = tokenizeWithValues("iffy");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[0].value, "iffy");
}

// ============== Integer Tests ==============

TEST(LexerTest, IntegerZero) {
  auto tokens = tokenizeWithValues("0");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TINTEGER));
  EXPECT_EQ(tokens[0].value, "0");
}

TEST(LexerTest, IntegerPositive) {
  auto tokens = tokenizeWithValues("42");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TINTEGER));
  EXPECT_EQ(tokens[0].value, "42");
}

TEST(LexerTest, IntegerLarge) {
  auto tokens = tokenizeWithValues("1234567890");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TINTEGER));
  EXPECT_EQ(tokens[0].value, "1234567890");
}

// ============== Double Tests ==============

TEST(LexerTest, DoubleSimple) {
  auto tokens = tokenizeWithValues("3.14");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TDOUBLE));
  EXPECT_EQ(tokens[0].value, "3.14");
}

TEST(LexerTest, DoubleNoFraction) {
  auto tokens = tokenizeWithValues("3.");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TDOUBLE));
  EXPECT_EQ(tokens[0].value, "3.");
}

TEST(LexerTest, DoubleZeroPrefix) {
  auto tokens = tokenizeWithValues("0.5");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, static_cast<int>(S_TDOUBLE));
  EXPECT_EQ(tokens[0].value, "0.5");
}

// ============== Operator Tests ==============

TEST(LexerTest, OperatorPlus) {
  auto tokens = tokenize("+");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TPLUS));
}

TEST(LexerTest, OperatorMinus) {
  auto tokens = tokenize("-");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TMINUS));
}

TEST(LexerTest, OperatorMul) {
  auto tokens = tokenize("*");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TMUL));
}

TEST(LexerTest, OperatorDiv) {
  auto tokens = tokenize("/");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TDIV));
}

TEST(LexerTest, OperatorEqual) {
  auto tokens = tokenize("=");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TEQUAL));
}

TEST(LexerTest, OperatorCompareEqual) {
  auto tokens = tokenize("==");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TCEQ));
}

TEST(LexerTest, OperatorNotEqual) {
  auto tokens = tokenize("!=");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TCNE));
}

TEST(LexerTest, OperatorLessThan) {
  auto tokens = tokenize("<");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TCLT));
}

TEST(LexerTest, OperatorLessEqual) {
  auto tokens = tokenize("<=");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TCLE));
}

TEST(LexerTest, OperatorGreaterThan) {
  auto tokens = tokenize(">");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TCGT));
}

TEST(LexerTest, OperatorGreaterEqual) {
  auto tokens = tokenize(">=");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TCGE));
}

TEST(LexerTest, OperatorArrow) {
  auto tokens = tokenize("->");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TARROW));
}

// ============== Punctuation Tests ==============

TEST(LexerTest, Parentheses) {
  auto tokens = tokenize("()");
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLPAREN));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TRPAREN));
}

TEST(LexerTest, Braces) {
  auto tokens = tokenize("{}");
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLBRACE));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TRBRACE));
}

TEST(LexerTest, Comma) {
  auto tokens = tokenize(",");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TCOMMA));
}

TEST(LexerTest, Colon) {
  auto tokens = tokenize(":");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TCOLON));
}

TEST(LexerTest, Dot) {
  auto tokens = tokenize(".");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TDOT));
}

// ============== Whitespace Tests ==============

TEST(LexerTest, WhitespaceIgnored) {
  auto tokens = tokenize("  \t\n  ");
  EXPECT_EQ(tokens.size(), 0);
}

TEST(LexerTest, TokensSeparatedByWhitespace) {
  auto tokens = tokenize("let x = 5");
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLET));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[2], static_cast<int>(S_TEQUAL));
  EXPECT_EQ(tokens[3], static_cast<int>(S_TINTEGER));
}

TEST(LexerTest, TokensSeparatedByNewline) {
  auto tokens = tokenize("let\nx\n=\n5");
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLET));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[2], static_cast<int>(S_TEQUAL));
  EXPECT_EQ(tokens[3], static_cast<int>(S_TINTEGER));
}

// ============== Complex Expression Tests ==============

TEST(LexerTest, VariableDeclaration) {
  auto tokens = tokenize("let x = 42");
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLET));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[2], static_cast<int>(S_TEQUAL));
  EXPECT_EQ(tokens[3], static_cast<int>(S_TINTEGER));
}

TEST(LexerTest, TypedVariableDeclaration) {
  auto tokens = tokenize("let x : int = 42");
  ASSERT_EQ(tokens.size(), 6);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLET));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[2], static_cast<int>(S_TCOLON));
  EXPECT_EQ(tokens[3], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[4], static_cast<int>(S_TEQUAL));
  EXPECT_EQ(tokens[5], static_cast<int>(S_TINTEGER));
}

TEST(LexerTest, FunctionDeclaration) {
  auto tokens = tokenize("let add(x: int, y: int): int = x + y");
  ASSERT_EQ(tokens.size(), 17);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLET));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TIDENTIFIER)); // add
  EXPECT_EQ(tokens[2], static_cast<int>(S_TLPAREN));
  EXPECT_EQ(tokens[3], static_cast<int>(S_TIDENTIFIER)); // x
  EXPECT_EQ(tokens[4], static_cast<int>(S_TCOLON));
  EXPECT_EQ(tokens[5], static_cast<int>(S_TIDENTIFIER)); // int
  EXPECT_EQ(tokens[6], static_cast<int>(S_TCOMMA));
}

TEST(LexerTest, FunctionCall) {
  auto tokens = tokenize("add(1, 2)");
  ASSERT_EQ(tokens.size(), 6);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TLPAREN));
  EXPECT_EQ(tokens[2], static_cast<int>(S_TINTEGER));
  EXPECT_EQ(tokens[3], static_cast<int>(S_TCOMMA));
  EXPECT_EQ(tokens[4], static_cast<int>(S_TINTEGER));
  EXPECT_EQ(tokens[5], static_cast<int>(S_TRPAREN));
}

TEST(LexerTest, IfExpression) {
  auto tokens = tokenize("if x > 0 then 1 else 0");
  ASSERT_EQ(tokens.size(), 8);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TIF));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[2], static_cast<int>(S_TCGT));
  EXPECT_EQ(tokens[3], static_cast<int>(S_TINTEGER));
  EXPECT_EQ(tokens[4], static_cast<int>(S_TTHEN));
  EXPECT_EQ(tokens[5], static_cast<int>(S_TINTEGER));
  EXPECT_EQ(tokens[6], static_cast<int>(S_TELSE));
  EXPECT_EQ(tokens[7], static_cast<int>(S_TINTEGER));
}

TEST(LexerTest, ArithmeticExpression) {
  auto tokens = tokenize("(a + b) * c / d - e");
  ASSERT_EQ(tokens.size(), 11);
  EXPECT_EQ(tokens[0], static_cast<int>(S_TLPAREN));
  EXPECT_EQ(tokens[1], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[2], static_cast<int>(S_TPLUS));
  EXPECT_EQ(tokens[3], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[4], static_cast<int>(S_TRPAREN));
  EXPECT_EQ(tokens[5], static_cast<int>(S_TMUL));
  EXPECT_EQ(tokens[6], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[7], static_cast<int>(S_TDIV));
  EXPECT_EQ(tokens[8], static_cast<int>(S_TIDENTIFIER));
  EXPECT_EQ(tokens[9], static_cast<int>(S_TMINUS));
  EXPECT_EQ(tokens[10], static_cast<int>(S_TIDENTIFIER));
}
