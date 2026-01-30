#ifndef POLANG_REPL_SESSION_HPP
#define POLANG_REPL_SESSION_HPP

#include <cstdint>
#include <memory>
#include <string>

class NBlock;
class TypeChecker;

namespace polang {
class JITSession;
} // namespace polang

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
  ReplSession() noexcept;
  ~ReplSession() noexcept;

  // Initialize LLVM - must be called before evaluate
  bool initialize();

  // Evaluate input and return result
  EvalResult evaluate(const std::string& input);

  // Check if input appears incomplete (needs more lines)
  static bool isInputIncomplete(const std::string& input) noexcept;

private:
  bool initialized = false;

  // Accumulated AST from previous successful evaluations
  std::unique_ptr<NBlock> accumulatedAst;

  // Persistent type checker across evaluations
  std::unique_ptr<TypeChecker> typeChecker;

  // Persistent JIT session across evaluations
  std::unique_ptr<polang::JITSession> jitSession;

  // Evaluation counter for unique entry function names
  int evalCounter = 0;
};

#endif // POLANG_REPL_SESSION_HPP
