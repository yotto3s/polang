#ifndef POLANG_ERROR_REPORTER_HPP
#define POLANG_ERROR_REPORTER_HPP

#include <functional>
#include <string>
#include <vector>

namespace polang {

/// Severity level for compiler errors.
enum class ErrorSeverity { Warning, Error, Fatal };

/// Represents a single compiler error with location information.
struct CompilerError {
  ErrorSeverity severity;
  std::string message;
  std::string filename;
  int line;
  int column;

  CompilerError(ErrorSeverity sev, const std::string& msg, int l = 0, int c = 0)
      : severity(sev), message(msg), line(l), column(c) {}

  /// Format the error as a human-readable string.
  std::string format() const;
};

/// Unified error reporter for the Polang compiler.
/// Provides a consistent way to report errors across all components.
class ErrorReporter {
public:
  using ErrorCallback = std::function<void(const CompilerError&)>;

  ErrorReporter() = default;

  /// Get the thread-local current error reporter.
  /// Returns nullptr if no reporter is set.
  static ErrorReporter* current();

  /// Set the current thread-local error reporter.
  static void setCurrent(ErrorReporter* reporter);

  /// Report an error with the given severity, message, and optional location.
  void report(ErrorSeverity severity, const std::string& message, int line = 0,
              int column = 0);

  /// Convenience methods for common error severities.
  void warning(const std::string& message, int line = 0, int column = 0);
  void error(const std::string& message, int line = 0, int column = 0);
  void fatal(const std::string& message, int line = 0, int column = 0);

  /// Set a callback to be invoked for each error reported.
  void setCallback(ErrorCallback callback);

  /// Set the current filename for error messages.
  void setFilename(const std::string& filename);

  /// Get all errors collected so far.
  const std::vector<CompilerError>& errors() const { return errors_; }

  /// Check if any errors have been reported.
  bool hasErrors() const;

  /// Check if any warnings have been reported.
  bool hasWarnings() const;

  /// Clear all collected errors.
  void clear();

private:
  std::vector<CompilerError> errors_;
  ErrorCallback callback_;
  std::string currentFilename_;
};

} // namespace polang

/// C-compatible function for the lexer to report errors.
/// This allows the lexer (which is C code) to call the error reporter.
extern "C" void polang_report_error(const char* message, int line, int column);

#endif // POLANG_ERROR_REPORTER_HPP
