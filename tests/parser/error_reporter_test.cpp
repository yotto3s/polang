#include <gtest/gtest.h>

#include "parser/error_reporter.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace polang;

// Helper class to capture stderr output
class StderrCapture {
public:
  StderrCapture() {
    original_stderr_ = stderr;
    captured_file_ = std::tmpfile();
    if (captured_file_) {
      stderr = captured_file_;
    }
  }

  ~StderrCapture() {
    stderr = original_stderr_;
    if (captured_file_) {
      std::fclose(captured_file_);
    }
  }

  std::string getCaptured() {
    if (!captured_file_) {
      return "";
    }
    std::fflush(captured_file_);
    std::rewind(captured_file_);
    std::string result;
    char buffer[256];
    while (std::fgets(buffer, sizeof(buffer), captured_file_)) {
      result += buffer;
    }
    return result;
  }

private:
  std::FILE* original_stderr_;
  std::FILE* captured_file_;
};

// ============== CompilerError Tests ==============

TEST(CompilerErrorTest, FormatWarning) {
  CompilerError err(ErrorSeverity::Warning, "test warning");
  std::string formatted = err.format();
  EXPECT_TRUE(formatted.find("warning:") != std::string::npos);
  EXPECT_TRUE(formatted.find("test warning") != std::string::npos);
}

TEST(CompilerErrorTest, FormatError) {
  CompilerError err(ErrorSeverity::Error, "test error");
  std::string formatted = err.format();
  EXPECT_TRUE(formatted.find("error:") != std::string::npos);
  EXPECT_TRUE(formatted.find("test error") != std::string::npos);
}

TEST(CompilerErrorTest, FormatFatal) {
  CompilerError err(ErrorSeverity::Fatal, "test fatal");
  std::string formatted = err.format();
  EXPECT_TRUE(formatted.find("fatal error:") != std::string::npos);
  EXPECT_TRUE(formatted.find("test fatal") != std::string::npos);
}

TEST(CompilerErrorTest, FormatWithLine) {
  CompilerError err(ErrorSeverity::Error, "test", 5);
  std::string formatted = err.format();
  EXPECT_TRUE(formatted.find("line 5") != std::string::npos);
}

TEST(CompilerErrorTest, FormatWithLineAndColumn) {
  CompilerError err(ErrorSeverity::Error, "test", 5, 10);
  std::string formatted = err.format();
  EXPECT_TRUE(formatted.find("line 5") != std::string::npos);
  EXPECT_TRUE(formatted.find("column 10") != std::string::npos);
}

TEST(CompilerErrorTest, FormatNoLocation) {
  CompilerError err(ErrorSeverity::Error, "test", 0, 0);
  std::string formatted = err.format();
  // Line 0 should not appear in output
  EXPECT_TRUE(formatted.find("line") == std::string::npos);
}

// ============== ErrorReporter Current/SetCurrent Tests ==============

TEST(ErrorReporterTest, InitialCurrentIsNull) {
  // Save any existing reporter
  ErrorReporter* saved = ErrorReporter::current();
  ErrorReporter::setCurrent(nullptr);

  EXPECT_EQ(ErrorReporter::current(), nullptr);

  // Restore
  ErrorReporter::setCurrent(saved);
}

TEST(ErrorReporterTest, SetAndGetCurrent) {
  ErrorReporter* saved = ErrorReporter::current();

  ErrorReporter reporter;
  ErrorReporter::setCurrent(&reporter);
  EXPECT_EQ(ErrorReporter::current(), &reporter);

  ErrorReporter::setCurrent(saved);
}

// ============== ErrorReporter Report Methods Tests ==============

TEST(ErrorReporterTest, ReportWarning) {
  StderrCapture capture;
  ErrorReporter reporter;

  reporter.warning("test warning", 1, 2);

  EXPECT_TRUE(reporter.hasWarnings());
  EXPECT_FALSE(reporter.hasErrors());
  EXPECT_EQ(reporter.errors().size(), 1);
  EXPECT_EQ(reporter.errors()[0].severity, ErrorSeverity::Warning);
  EXPECT_EQ(reporter.errors()[0].message, "test warning");
  EXPECT_EQ(reporter.errors()[0].line, 1);
  EXPECT_EQ(reporter.errors()[0].column, 2);
}

TEST(ErrorReporterTest, ReportError) {
  StderrCapture capture;
  ErrorReporter reporter;

  reporter.error("test error", 3, 4);

  EXPECT_FALSE(reporter.hasWarnings());
  EXPECT_TRUE(reporter.hasErrors());
  EXPECT_EQ(reporter.errors().size(), 1);
  EXPECT_EQ(reporter.errors()[0].severity, ErrorSeverity::Error);
}

TEST(ErrorReporterTest, ReportFatal) {
  StderrCapture capture;
  ErrorReporter reporter;

  reporter.fatal("test fatal", 5, 6);

  EXPECT_TRUE(reporter.hasErrors()); // Fatal counts as error
  EXPECT_EQ(reporter.errors().size(), 1);
  EXPECT_EQ(reporter.errors()[0].severity, ErrorSeverity::Fatal);
}

TEST(ErrorReporterTest, MultipleErrors) {
  StderrCapture capture;
  ErrorReporter reporter;

  reporter.warning("warn1");
  reporter.error("err1");
  reporter.warning("warn2");
  reporter.error("err2");

  EXPECT_TRUE(reporter.hasWarnings());
  EXPECT_TRUE(reporter.hasErrors());
  EXPECT_EQ(reporter.errors().size(), 4);
}

// ============== ErrorReporter Clear Tests ==============

TEST(ErrorReporterTest, ClearErrors) {
  StderrCapture capture;
  ErrorReporter reporter;

  reporter.error("error1");
  reporter.warning("warning1");
  EXPECT_TRUE(reporter.hasErrors());
  EXPECT_TRUE(reporter.hasWarnings());

  reporter.clear();

  EXPECT_FALSE(reporter.hasErrors());
  EXPECT_FALSE(reporter.hasWarnings());
  EXPECT_TRUE(reporter.errors().empty());
}

// ============== ErrorReporter Callback Tests ==============

TEST(ErrorReporterTest, CallbackInvoked) {
  StderrCapture capture;
  ErrorReporter reporter;
  std::vector<CompilerError> captured_errors;

  reporter.setCallback([&captured_errors](const CompilerError& err) {
    captured_errors.push_back(err);
  });

  reporter.error("test error");
  reporter.warning("test warning");

  EXPECT_EQ(captured_errors.size(), 2);
  EXPECT_EQ(captured_errors[0].message, "test error");
  EXPECT_EQ(captured_errors[1].message, "test warning");
}

// ============== ErrorReporter Filename Tests ==============

TEST(ErrorReporterTest, SetFilename) {
  StderrCapture capture;
  ErrorReporter reporter;

  reporter.setFilename("test.po");
  reporter.error("test error");

  EXPECT_EQ(reporter.errors()[0].filename, "test.po");
}

// ============== ErrorReporter Output to stderr Tests ==============

TEST(ErrorReporterTest, OutputToStderr) {
  StderrCapture capture;
  ErrorReporter reporter;

  reporter.error("stderr test", 10, 20);

  std::string output = capture.getCaptured();
  EXPECT_TRUE(output.find("error:") != std::string::npos);
  EXPECT_TRUE(output.find("stderr test") != std::string::npos);
  EXPECT_TRUE(output.find("line 10") != std::string::npos);
  EXPECT_TRUE(output.find("column 20") != std::string::npos);
}

// ============== polang_report_error Tests ==============

TEST(ErrorReporterTest, CReportErrorWithReporter) {
  StderrCapture capture;
  ErrorReporter reporter;
  ErrorReporter* saved = ErrorReporter::current();
  ErrorReporter::setCurrent(&reporter);

  polang_report_error("test_token", 5, 10);

  EXPECT_TRUE(reporter.hasErrors());
  EXPECT_EQ(reporter.errors().size(), 1);
  EXPECT_TRUE(reporter.errors()[0].message.find("test_token") !=
              std::string::npos);
  EXPECT_EQ(reporter.errors()[0].line, 5);
  EXPECT_EQ(reporter.errors()[0].column, 10);

  ErrorReporter::setCurrent(saved);
}

TEST(ErrorReporterTest, CReportErrorWithoutReporter) {
  StderrCapture capture;
  ErrorReporter* saved = ErrorReporter::current();
  ErrorReporter::setCurrent(nullptr);

  polang_report_error("fallback_token", 1, 2);

  std::string output = capture.getCaptured();
  EXPECT_TRUE(output.find("unknown token") != std::string::npos);
  EXPECT_TRUE(output.find("fallback_token") != std::string::npos);

  ErrorReporter::setCurrent(saved);
}
