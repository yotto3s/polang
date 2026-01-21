#ifndef POLANG_ERROR_REPORTER_HPP
#define POLANG_ERROR_REPORTER_HPP

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace polang {

/// Severity level for compiler errorList.
enum class ErrorSeverity { Warning, Error, Fatal };

/// Represents a single compiler error with location information.
struct CompilerError {
  ErrorSeverity severity;
  std::string message;
  std::string filename;
  int line;
  int column;

  CompilerError(ErrorSeverity sev, std::string msg, int l = 0, int c = 0)
      : severity(sev), message(std::move(msg)), line(l), column(c) {}

  /// Format the error as a human-readable string.
  [[nodiscard]] std::string format() const;
};

/// Unified error reporter for the Polang compiler.
/// Provides a consistent way to report errorList across all components.
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

  /// Set a errorCallback to be invoked for each error reported.
  void setCallback(ErrorCallback cb);

  /// Set the current filename for error messages.
  void setFilename(const std::string& filename);

  /// Get all errorList collected so far.
  [[nodiscard]] const std::vector<CompilerError>& errors() const {
    return errorList;
  }

  /// Check if any errorList have been reported.
  [[nodiscard]] bool hasErrors() const;

  /// Check if any warnings have been reported.
  [[nodiscard]] bool hasWarnings() const;

  /// Clear all collected errorList.
  void clear();

private:
  std::vector<CompilerError> errorList;
  ErrorCallback errorCallback;
  std::string currentFilename;
};

// Error message formatting helpers.
// These provide consistent formatting for common error types.

/// Format a type mismatch error message.
/// @param context Description of where the mismatch occurred (e.g., "in '+'")
/// @param expected The expected type
/// @param actual The actual type found
[[nodiscard]] inline std::string formatTypeMismatch(const std::string& context,
                                                    const std::string& expected,
                                                    const std::string& actual) {
  return "Type mismatch " + context + ": expected " + expected + ", got " +
         actual;
}

/// Format an argument count error message.
/// @param funcName The function name
/// @param expected The expected number of arguments
/// @param actual The actual number of arguments provided
[[nodiscard]] inline std::string
formatArgCountError(const std::string& funcName, size_t expected,
                    size_t actual) {
  return "Function '" + funcName + "' expects " + std::to_string(expected) +
         " argument(s), got " + std::to_string(actual);
}

/// Format a variable declaration type error message.
/// @param varName The variable name
/// @param declaredType The declared type annotation
/// @param actualType The actual type of the initializer
[[nodiscard]] inline std::string
formatVarDeclTypeError(const std::string& varName,
                       const std::string& declaredType,
                       const std::string& actualType) {
  return "Variable '" + varName + "' declared as " + declaredType +
         " but initialized with " + actualType + " (no implicit conversion)";
}

/// Format an undeclared variable error message.
/// @param varName The variable name
[[nodiscard]] inline std::string
formatUndeclaredVar(const std::string& varName) {
  return "Undeclared variable: " + varName;
}

/// Format a function return type error message.
/// @param funcName The function name
/// @param declaredType The declared return type
/// @param actualType The actual type returned
[[nodiscard]] inline std::string
formatFuncReturnTypeError(const std::string& funcName,
                          const std::string& declaredType,
                          const std::string& actualType) {
  return "Function '" + funcName + "' declared to return " + declaredType +
         " but body has type " + actualType;
}

} // namespace polang

/// C-compatible function for the lexer to report errorList.
/// This allows the lexer (which is C code) to call the error reporter.
extern "C" void polang_report_error(const char* message, int line, int column);

#endif // POLANG_ERROR_REPORTER_HPP
