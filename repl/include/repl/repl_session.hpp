#ifndef POLANG_REPL_SESSION_HPP
#define POLANG_REPL_SESSION_HPP

#include <cstdint>
#include <string>

// Result of evaluating an expression
struct EvalResult {
  bool success;
  bool hasValue;
  int64_t rawValue;
  std::string type; // "int", "double", "bool", "void"
  std::string errorMessage;

  static EvalResult ok() { return {true, false, 0, "void", ""}; }

  static EvalResult value(int64_t val, const std::string& t) {
    return {true, true, val, t, ""};
  }

  static EvalResult error(const std::string& msg) {
    return {false, false, 0, "", msg};
  }
};

// Manages persistent state for the REPL session
class ReplSession {
public:
  ReplSession();
  ~ReplSession();

  // Initialize LLVM - must be called before evaluate
  bool initialize();

  // Evaluate input and return result
  EvalResult evaluate(const std::string& input);

  // Check if input appears incomplete (needs more lines)
  static bool isInputIncomplete(const std::string& input);

private:
  bool initialized_ = false;

  // Accumulated source code from previous successful evaluations
  std::string accumulatedCode_;
};

#endif // POLANG_REPL_SESSION_HPP
