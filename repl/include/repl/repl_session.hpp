#ifndef POLANG_REPL_SESSION_HPP
#define POLANG_REPL_SESSION_HPP

#include "compiler/jit_result.hpp"

#include <memory>
#include <string>
#include <variant>

class NBlock;
class TypeChecker;

namespace polang {
class JITSession;
struct CompiledSymbols;
} // namespace polang

// Result of evaluating an expression
struct EvalResult {
  bool success;
  polang::JitResult value;
  std::string type; // "i64", "f64", "f32", "bool", "void"
  std::string errorMessage;

  [[nodiscard]] bool hasValue() const noexcept {
    return !std::holds_alternative<std::monostate>(value);
  }

  static EvalResult ok() { return {true, std::monostate{}, "void", ""}; }

  static EvalResult withValue(polang::JitResult val, const std::string& t) {
    return {true, val, t, ""};
  }

  static EvalResult error(const std::string& msg) {
    return {false, std::monostate{}, "", msg};
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

  // Registry of previously compiled symbols for incremental compilation
  std::unique_ptr<polang::CompiledSymbols> compiledSymbols;

  // Persistent type checker across evaluations
  std::unique_ptr<TypeChecker> typeChecker;

  // Persistent JIT session across evaluations
  std::unique_ptr<polang::JITSession> jitSession;

  // Evaluation counter for unique entry function names
  int evalCounter = 0;

  // Update compiled symbols after a successful evaluation
  void updateCompiledSymbols(NBlock& newAst);
};

#endif // POLANG_REPL_SESSION_HPP
