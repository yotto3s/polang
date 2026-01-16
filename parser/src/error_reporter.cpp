#include "parser/error_reporter.hpp"

#include <cstdio>
#include <sstream>

namespace polang {

// Thread-local pointer to the current error reporter
static thread_local ErrorReporter* currentReporter = nullptr;

std::string CompilerError::format() const {
  std::ostringstream oss;

  // Severity prefix
  switch (severity) {
  case ErrorSeverity::Warning:
    oss << "warning: ";
    break;
  case ErrorSeverity::Error:
    oss << "error: ";
    break;
  case ErrorSeverity::Fatal:
    oss << "fatal error: ";
    break;
  }

  // Message
  oss << message;

  // Location (if available)
  if (line > 0) {
    oss << " at line " << line;
    if (column > 0) {
      oss << ", column " << column;
    }
  }

  return oss.str();
}

ErrorReporter* ErrorReporter::current() { return currentReporter; }

void ErrorReporter::setCurrent(ErrorReporter* reporter) {
  currentReporter = reporter;
}

void ErrorReporter::report(ErrorSeverity severity, const std::string& message,
                           int line, int column) {
  CompilerError err(severity, message, line, column);
  err.filename = currentFilename_;
  errors_.push_back(err);

  // Invoke callback if set
  if (callback_) {
    callback_(err);
  }

  // Default behavior: print to stderr (using fprintf for C compatibility)
  std::fprintf(stderr, "%s\n", err.format().c_str());
}

void ErrorReporter::warning(const std::string& message, int line, int column) {
  report(ErrorSeverity::Warning, message, line, column);
}

void ErrorReporter::error(const std::string& message, int line, int column) {
  report(ErrorSeverity::Error, message, line, column);
}

void ErrorReporter::fatal(const std::string& message, int line, int column) {
  report(ErrorSeverity::Fatal, message, line, column);
}

void ErrorReporter::setCallback(ErrorCallback callback) {
  callback_ = std::move(callback);
}

void ErrorReporter::setFilename(const std::string& filename) {
  currentFilename_ = filename;
}

bool ErrorReporter::hasErrors() const {
  for (const auto& err : errors_) {
    if (err.severity == ErrorSeverity::Error ||
        err.severity == ErrorSeverity::Fatal) {
      return true;
    }
  }
  return false;
}

bool ErrorReporter::hasWarnings() const {
  for (const auto& err : errors_) {
    if (err.severity == ErrorSeverity::Warning) {
      return true;
    }
  }
  return false;
}

void ErrorReporter::clear() { errors_.clear(); }

} // namespace polang

// C-compatible function for lexer
extern "C" void polang_report_error(const char* message, int line, int column) {
  auto* reporter = polang::ErrorReporter::current();
  if (reporter) {
    reporter->error(std::string("unknown token '") + message + "'", line,
                    column);
  } else {
    // Fallback if no reporter is set (using fprintf for C compatibility)
    std::fprintf(stderr, "error: unknown token '%s' at line %d, column %d\n",
                 message, line, column);
  }
}
