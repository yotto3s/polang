// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
#include <string>
#include <vector>

// Parser headers (includes LLVM via node.hpp)
// clang-format off
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison union types
// clang-format on

// Flex buffer type and functions
typedef struct yy_buffer_state* YY_BUFFER_STATE;
extern YY_BUFFER_STATE yy_scan_string(const char* str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
extern int yylex();
extern YYSTYPE yylval;

// Helper to get all tokens from a source string
std::vector<int> tokenize(const std::string& source) {
  std::vector<int> tokens;
  YY_BUFFER_STATE buffer = yy_scan_string(source.c_str());
  int token;
  while ((token = yylex()) != 0) {
    tokens.push_back(token);
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
  int token;
  while ((token = yylex()) != 0) {
    TokenInfo info;
    info.token = token;
    // For tokens that store string values
    if (token == TIDENTIFIER || token == TINTEGER || token == TDOUBLE) {
      if (yylval.string) {
        info.value = *yylval.string;
        delete yylval.string;
        yylval.string = nullptr;
      }
    }
    tokens.push_back(info);
  }
  yy_delete_buffer(buffer);
  return tokens;
}

// ============== Keyword Tests ==============

TEST(LexerTest, KeywordLet) {
  auto tokens = tokenize("let");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TLET);
}

TEST(LexerTest, KeywordFun) {
  auto tokens = tokenize("fun");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TFUN);
}

TEST(LexerTest, KeywordIn) {
  auto tokens = tokenize("in");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TIN);
}

TEST(LexerTest, KeywordIf) {
  auto tokens = tokenize("if");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TIF);
}

TEST(LexerTest, KeywordThen) {
  auto tokens = tokenize("then");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TTHEN);
}

TEST(LexerTest, KeywordElse) {
  auto tokens = tokenize("else");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TELSE);
}

// ============== Identifier Tests ==============

TEST(LexerTest, SimpleIdentifier) {
  auto tokens = tokenizeWithValues("foo");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TIDENTIFIER);
  EXPECT_EQ(tokens[0].value, "foo");
}

TEST(LexerTest, IdentifierWithUnderscore) {
  auto tokens = tokenizeWithValues("my_var");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TIDENTIFIER);
  EXPECT_EQ(tokens[0].value, "my_var");
}

TEST(LexerTest, IdentifierStartingWithUnderscore) {
  auto tokens = tokenizeWithValues("_private");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TIDENTIFIER);
  EXPECT_EQ(tokens[0].value, "_private");
}

TEST(LexerTest, IdentifierWithNumbers) {
  auto tokens = tokenizeWithValues("var123");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TIDENTIFIER);
  EXPECT_EQ(tokens[0].value, "var123");
}

TEST(LexerTest, KeywordPrefixIdentifier) {
  // "letter" starts with "let" but should be identifier
  auto tokens = tokenizeWithValues("letter");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TIDENTIFIER);
  EXPECT_EQ(tokens[0].value, "letter");
}

TEST(LexerTest, IfPrefixIdentifier) {
  // "iffy" starts with "if" but should be identifier
  auto tokens = tokenizeWithValues("iffy");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TIDENTIFIER);
  EXPECT_EQ(tokens[0].value, "iffy");
}

// ============== Integer Tests ==============

TEST(LexerTest, IntegerZero) {
  auto tokens = tokenizeWithValues("0");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TINTEGER);
  EXPECT_EQ(tokens[0].value, "0");
}

TEST(LexerTest, IntegerPositive) {
  auto tokens = tokenizeWithValues("42");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TINTEGER);
  EXPECT_EQ(tokens[0].value, "42");
}

TEST(LexerTest, IntegerLarge) {
  auto tokens = tokenizeWithValues("1234567890");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TINTEGER);
  EXPECT_EQ(tokens[0].value, "1234567890");
}

// ============== Double Tests ==============

TEST(LexerTest, DoubleSimple) {
  auto tokens = tokenizeWithValues("3.14");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TDOUBLE);
  EXPECT_EQ(tokens[0].value, "3.14");
}

TEST(LexerTest, DoubleNoFraction) {
  auto tokens = tokenizeWithValues("3.");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TDOUBLE);
  EXPECT_EQ(tokens[0].value, "3.");
}

TEST(LexerTest, DoubleZeroPrefix) {
  auto tokens = tokenizeWithValues("0.5");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].token, TDOUBLE);
  EXPECT_EQ(tokens[0].value, "0.5");
}

// ============== Operator Tests ==============

TEST(LexerTest, OperatorPlus) {
  auto tokens = tokenize("+");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TPLUS);
}

TEST(LexerTest, OperatorMinus) {
  auto tokens = tokenize("-");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TMINUS);
}

TEST(LexerTest, OperatorMul) {
  auto tokens = tokenize("*");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TMUL);
}

TEST(LexerTest, OperatorDiv) {
  auto tokens = tokenize("/");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TDIV);
}

TEST(LexerTest, OperatorEqual) {
  auto tokens = tokenize("=");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TEQUAL);
}

TEST(LexerTest, OperatorCompareEqual) {
  auto tokens = tokenize("==");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TCEQ);
}

TEST(LexerTest, OperatorNotEqual) {
  auto tokens = tokenize("!=");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TCNE);
}

TEST(LexerTest, OperatorLessThan) {
  auto tokens = tokenize("<");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TCLT);
}

TEST(LexerTest, OperatorLessEqual) {
  auto tokens = tokenize("<=");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TCLE);
}

TEST(LexerTest, OperatorGreaterThan) {
  auto tokens = tokenize(">");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TCGT);
}

TEST(LexerTest, OperatorGreaterEqual) {
  auto tokens = tokenize(">=");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TCGE);
}

TEST(LexerTest, OperatorArrow) {
  auto tokens = tokenize("->");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TARROW);
}

// ============== Punctuation Tests ==============

TEST(LexerTest, Parentheses) {
  auto tokens = tokenize("()");
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], TLPAREN);
  EXPECT_EQ(tokens[1], TRPAREN);
}

TEST(LexerTest, Braces) {
  auto tokens = tokenize("{}");
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], TLBRACE);
  EXPECT_EQ(tokens[1], TRBRACE);
}

TEST(LexerTest, Comma) {
  auto tokens = tokenize(",");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TCOMMA);
}

TEST(LexerTest, Colon) {
  auto tokens = tokenize(":");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TCOLON);
}

TEST(LexerTest, Dot) {
  auto tokens = tokenize(".");
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], TDOT);
}

// ============== Whitespace Tests ==============

TEST(LexerTest, WhitespaceIgnored) {
  auto tokens = tokenize("  \t\n  ");
  EXPECT_EQ(tokens.size(), 0);
}

TEST(LexerTest, TokensSeparatedByWhitespace) {
  auto tokens = tokenize("let x = 5");
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], TLET);
  EXPECT_EQ(tokens[1], TIDENTIFIER);
  EXPECT_EQ(tokens[2], TEQUAL);
  EXPECT_EQ(tokens[3], TINTEGER);
}

TEST(LexerTest, TokensSeparatedByNewline) {
  auto tokens = tokenize("let\nx\n=\n5");
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], TLET);
  EXPECT_EQ(tokens[1], TIDENTIFIER);
  EXPECT_EQ(tokens[2], TEQUAL);
  EXPECT_EQ(tokens[3], TINTEGER);
}

// ============== Complex Expression Tests ==============

TEST(LexerTest, VariableDeclaration) {
  auto tokens = tokenize("let x = 42");
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], TLET);
  EXPECT_EQ(tokens[1], TIDENTIFIER);
  EXPECT_EQ(tokens[2], TEQUAL);
  EXPECT_EQ(tokens[3], TINTEGER);
}

TEST(LexerTest, TypedVariableDeclaration) {
  auto tokens = tokenize("let x : int = 42");
  ASSERT_EQ(tokens.size(), 6);
  EXPECT_EQ(tokens[0], TLET);
  EXPECT_EQ(tokens[1], TIDENTIFIER);
  EXPECT_EQ(tokens[2], TCOLON);
  EXPECT_EQ(tokens[3], TIDENTIFIER);
  EXPECT_EQ(tokens[4], TEQUAL);
  EXPECT_EQ(tokens[5], TINTEGER);
}

TEST(LexerTest, FunctionDeclaration) {
  auto tokens = tokenize("let add(x: int, y: int): int = x + y");
  ASSERT_EQ(tokens.size(), 17);
  EXPECT_EQ(tokens[0], TLET);
  EXPECT_EQ(tokens[1], TIDENTIFIER); // add
  EXPECT_EQ(tokens[2], TLPAREN);
  EXPECT_EQ(tokens[3], TIDENTIFIER); // x
  EXPECT_EQ(tokens[4], TCOLON);
  EXPECT_EQ(tokens[5], TIDENTIFIER); // int
  EXPECT_EQ(tokens[6], TCOMMA);
}

TEST(LexerTest, FunctionCall) {
  auto tokens = tokenize("add(1, 2)");
  ASSERT_EQ(tokens.size(), 6);
  EXPECT_EQ(tokens[0], TIDENTIFIER);
  EXPECT_EQ(tokens[1], TLPAREN);
  EXPECT_EQ(tokens[2], TINTEGER);
  EXPECT_EQ(tokens[3], TCOMMA);
  EXPECT_EQ(tokens[4], TINTEGER);
  EXPECT_EQ(tokens[5], TRPAREN);
}

TEST(LexerTest, IfExpression) {
  auto tokens = tokenize("if x > 0 then 1 else 0");
  ASSERT_EQ(tokens.size(), 8);
  EXPECT_EQ(tokens[0], TIF);
  EXPECT_EQ(tokens[1], TIDENTIFIER);
  EXPECT_EQ(tokens[2], TCGT);
  EXPECT_EQ(tokens[3], TINTEGER);
  EXPECT_EQ(tokens[4], TTHEN);
  EXPECT_EQ(tokens[5], TINTEGER);
  EXPECT_EQ(tokens[6], TELSE);
  EXPECT_EQ(tokens[7], TINTEGER);
}

TEST(LexerTest, ArithmeticExpression) {
  auto tokens = tokenize("(a + b) * c / d - e");
  ASSERT_EQ(tokens.size(), 11);
  EXPECT_EQ(tokens[0], TLPAREN);
  EXPECT_EQ(tokens[1], TIDENTIFIER);
  EXPECT_EQ(tokens[2], TPLUS);
  EXPECT_EQ(tokens[3], TIDENTIFIER);
  EXPECT_EQ(tokens[4], TRPAREN);
  EXPECT_EQ(tokens[5], TMUL);
  EXPECT_EQ(tokens[6], TIDENTIFIER);
  EXPECT_EQ(tokens[7], TDIV);
  EXPECT_EQ(tokens[8], TIDENTIFIER);
  EXPECT_EQ(tokens[9], TMINUS);
  EXPECT_EQ(tokens[10], TIDENTIFIER);
}
