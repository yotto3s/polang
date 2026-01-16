// Include gtest first to avoid conflicts with LLVM headers
#include <gtest/gtest.h>

// Standard library
#include <cstdio>
#include <regex>
#include <sstream>
#include <string>

// clang-format off
// Parser headers
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison union types
#include "parser/parser_api.hpp"
// clang-format on

// Helper class to capture stderr output
class StderrCapture {
public:
  StderrCapture() {
    // Save original stderr
    original_stderr_ = stderr;
    // Create a temporary file to capture stderr
    captured_file_ = tmpfile();
    if (captured_file_) {
      // Redirect stderr to the temporary file
      stderr = captured_file_;
    }
  }

  ~StderrCapture() {
    // Restore original stderr
    stderr = original_stderr_;
    if (captured_file_) {
      fclose(captured_file_);
    }
  }

  std::string getCaptured() {
    if (!captured_file_) {
      return "";
    }
    // Flush any pending output
    fflush(captured_file_);
    // Read from the beginning of the temp file
    rewind(captured_file_);
    std::string result;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), captured_file_)) {
      result += buffer;
    }
    return result;
  }

private:
  FILE* original_stderr_;
  FILE* captured_file_;
};

// ============== Syntax Error Location Tests ==============

TEST(ErrorTest, SyntaxErrorAtEndOfLine) {
  StderrCapture capture;
  NBlock* block = polang_parse("let x =");
  std::string error = capture.getCaptured();

  // Error should mention line 1
  EXPECT_TRUE(error.find("line 1") != std::string::npos ||
              error.find("line 2") != std::string::npos)
      << "Error message should include line number. Got: " << error;
  EXPECT_TRUE(error.find("syntax error") != std::string::npos)
      << "Error message should mention 'syntax error'. Got: " << error;
}

TEST(ErrorTest, SyntaxErrorOnSecondLine) {
  StderrCapture capture;
  NBlock* block = polang_parse("let x: int = 5\nlet y =");
  std::string error = capture.getCaptured();

  // Error should be on line 2
  EXPECT_TRUE(error.find("line 2") != std::string::npos)
      << "Error should be on line 2. Got: " << error;
}

TEST(ErrorTest, SyntaxErrorOnThirdLine) {
  StderrCapture capture;
  NBlock* block = polang_parse("let x: int = 1\nlet y: int = 2\nlet z =");
  std::string error = capture.getCaptured();

  // Error should be on line 3
  EXPECT_TRUE(error.find("line 3") != std::string::npos)
      << "Error should be on line 3. Got: " << error;
}

TEST(ErrorTest, SyntaxErrorColumnPosition) {
  StderrCapture capture;
  NBlock* block = polang_parse("let longname: int = )");
  std::string error = capture.getCaptured();

  // Error should be around column 21 (where ')' appears)
  EXPECT_TRUE(error.find("column 21") != std::string::npos)
      << "Error should be at column 21. Got: " << error;
}

TEST(ErrorTest, SyntaxErrorMissingThen) {
  StderrCapture capture;
  NBlock* block = polang_parse("if 1 2 else 3");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("syntax error") != std::string::npos)
      << "Should report syntax error. Got: " << error;
  EXPECT_TRUE(error.find("line 1") != std::string::npos)
      << "Should report line 1. Got: " << error;
}

TEST(ErrorTest, SyntaxErrorMissingElse) {
  StderrCapture capture;
  NBlock* block = polang_parse("if 1 then 2");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("syntax error") != std::string::npos)
      << "Should report syntax error. Got: " << error;
}

TEST(ErrorTest, SyntaxErrorUnclosedParen) {
  StderrCapture capture;
  NBlock* block = polang_parse("(1 + 2");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("syntax error") != std::string::npos)
      << "Should report syntax error for unclosed paren. Got: " << error;
}

TEST(ErrorTest, SyntaxErrorMismatchedParen) {
  StderrCapture capture;
  NBlock* block = polang_parse("(1 + 2))");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("syntax error") != std::string::npos)
      << "Should report syntax error for mismatched paren. Got: " << error;
}

// ============== Unknown Token Error Tests ==============

TEST(ErrorTest, UnknownTokenAtStart) {
  StderrCapture capture;
  NBlock* block = polang_parse("@");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("unknown token") != std::string::npos)
      << "Should report unknown token. Got: " << error;
  EXPECT_TRUE(error.find("'@'") != std::string::npos)
      << "Should show the unknown token '@'. Got: " << error;
  EXPECT_TRUE(error.find("line 1") != std::string::npos)
      << "Should report line 1. Got: " << error;
  EXPECT_TRUE(error.find("column 1") != std::string::npos)
      << "Should report column 1. Got: " << error;
}

TEST(ErrorTest, UnknownTokenInMiddle) {
  StderrCapture capture;
  NBlock* block = polang_parse("let x = @");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("unknown token") != std::string::npos)
      << "Should report unknown token. Got: " << error;
  EXPECT_TRUE(error.find("'@'") != std::string::npos)
      << "Should show the unknown token '@'. Got: " << error;
  EXPECT_TRUE(error.find("column 9") != std::string::npos)
      << "Should report column 9. Got: " << error;
}

TEST(ErrorTest, UnknownTokenOnSecondLine) {
  StderrCapture capture;
  NBlock* block = polang_parse("let x: int = 5\n@");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("unknown token") != std::string::npos)
      << "Should report unknown token. Got: " << error;
  EXPECT_TRUE(error.find("line 2") != std::string::npos)
      << "Should report line 2. Got: " << error;
  EXPECT_TRUE(error.find("column 1") != std::string::npos)
      << "Should report column 1. Got: " << error;
}

TEST(ErrorTest, UnknownTokenWithOffset) {
  StderrCapture capture;
  NBlock* block = polang_parse("let x: int = 5\n    @");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("unknown token") != std::string::npos)
      << "Should report unknown token. Got: " << error;
  EXPECT_TRUE(error.find("line 2") != std::string::npos)
      << "Should report line 2. Got: " << error;
  EXPECT_TRUE(error.find("column 5") != std::string::npos)
      << "Should report column 5 (after 4 spaces). Got: " << error;
}

TEST(ErrorTest, UnknownTokenDifferentChars) {
  // Test various unknown characters
  const char* unknown_chars[] = {"#", "$", "`", "~", "\\", "^"};

  for (const char* ch : unknown_chars) {
    StderrCapture capture;
    NBlock* block = polang_parse(ch);
    std::string error = capture.getCaptured();

    std::string expected_token = std::string("'") + ch + "'";
    EXPECT_TRUE(error.find("unknown token") != std::string::npos)
        << "Should report unknown token for " << ch << ". Got: " << error;
  }
}

// ============== Multiple Errors Tests ==============

TEST(ErrorTest, MultipleErrorsOnDifferentLines) {
  // First error terminates lexing, so we only get one error
  StderrCapture capture;
  NBlock* block = polang_parse("let x: int = @\nlet y: int = #");
  std::string error = capture.getCaptured();

  // Should get at least the first error
  EXPECT_TRUE(error.find("line 1") != std::string::npos ||
              error.find("unknown token") != std::string::npos)
      << "Should report at least first error. Got: " << error;
}

// ============== Error Recovery Tests ==============

TEST(ErrorTest, ValidCodeAfterError) {
  // Parse valid code to ensure parser state is properly reset
  StderrCapture capture1;
  polang_parse("@"); // This should error

  // Now parse valid code - it should work
  StderrCapture capture2;
  NBlock* block = polang_parse("let x: int = 5");
  std::string error2 = capture2.getCaptured();

  EXPECT_TRUE(error2.empty() || error2.find("error") == std::string::npos)
      << "Valid code should not produce errors. Got: " << error2;
  EXPECT_NE(block, nullptr) << "Valid code should parse successfully";
}

TEST(ErrorTest, LocationResetBetweenParses) {
  // First parse with error on line 2
  {
    StderrCapture capture;
    polang_parse("let x: int = 5\n@");
    std::string error = capture.getCaptured();
    EXPECT_TRUE(error.find("line 2") != std::string::npos)
        << "First parse error should be on line 2. Got: " << error;
  }

  // Second parse with error on line 1 - should reset location
  {
    StderrCapture capture;
    polang_parse("@");
    std::string error = capture.getCaptured();
    EXPECT_TRUE(error.find("line 1") != std::string::npos)
        << "Second parse error should be on line 1 (reset). Got: " << error;
  }
}

// ============== Complex Error Scenarios ==============

TEST(ErrorTest, ErrorInFunctionDeclaration) {
  StderrCapture capture;
  NBlock* block = polang_parse("let add (x: int) (y: int) = @");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("unknown token") != std::string::npos)
      << "Should report unknown token. Got: " << error;
}

TEST(ErrorTest, ErrorInNestedExpression) {
  StderrCapture capture;
  NBlock* block = polang_parse("let x = (1 + (2 * @))");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("unknown token") != std::string::npos)
      << "Should report unknown token. Got: " << error;
}

TEST(ErrorTest, ErrorInIfExpression) {
  StderrCapture capture;
  NBlock* block = polang_parse("if @ then 1 else 0");
  std::string error = capture.getCaptured();

  EXPECT_TRUE(error.find("unknown token") != std::string::npos)
      << "Should report unknown token. Got: " << error;
  EXPECT_TRUE(error.find("column 4") != std::string::npos)
      << "Error should be at column 4. Got: " << error;
}
