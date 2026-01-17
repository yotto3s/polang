// Suppress warnings from LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "repl/repl_session.hpp"
#include "repl/input_checker.hpp"

#include "compiler/mlir_codegen.hpp"
#include "parser/node.hpp"
#include "parser/parser_api.hpp"
#include "parser/type_checker.hpp"

#include <llvm/Support/TargetSelect.h>

#pragma GCC diagnostic pop

#include <iostream>

using namespace llvm;

ReplSession::ReplSession() noexcept = default;
ReplSession::~ReplSession() noexcept = default;

bool ReplSession::initialize() {
  if (initialized) {
    return true;
  }

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  initialized = true;
  return true;
}

bool ReplSession::isInputIncomplete(const std::string& input) noexcept {
  return InputChecker::isInputIncomplete(input);
}

EvalResult ReplSession::evaluate(const std::string& input) {
  if (!initialized) {
    return EvalResult::error("REPL session not initialized");
  }

  // Combine accumulated code with new input
  const std::string fullCode = accumulatedCode + input;

  // Parse the combined input
  auto ast = polang_parse(fullCode);
  if (ast == nullptr) {
    return EvalResult::error("Parse error");
  }

  // Type check the combined code
  const auto errors = polang_check_types(*ast);
  if (!errors.empty()) {
    std::string errMsg;
    for (const auto& err : errors) {
      errMsg += err.message;
      if (&err != &errors.back()) {
        errMsg += "\n";
      }
    }
    return EvalResult::error(errMsg);
  }

  // Check if the last statement is an expression (should print result)
  // vs a declaration (shouldn't print result)
  bool lastIsExpression = false;
  std::string resultType = "void";
  if (!ast->statements.empty()) {
    const NStatement* lastStmt = ast->statements.back().get();
    // NExpressionStatement wraps expressions as statements
    if (dynamic_cast<const NExpressionStatement*>(lastStmt) != nullptr) {
      lastIsExpression = true;
      // Type will be resolved from MLIR after type inference
    }
  }

  // Generate code using MLIR backend - always emit type variables
  polang::MLIRCodeGenContext codegenCtx;

  if (!codegenCtx.generateCode(*ast, true)) {
    std::cerr << "MLIR generation failed: " << codegenCtx.getError() << "\n";
    return EvalResult::error("Code generation failed");
  }

  // Run type inference to resolve type variables
  if (!codegenCtx.runTypeInference()) {
    std::cerr << "Type inference failed: " << codegenCtx.getError() << "\n";
    return EvalResult::error("Type inference failed");
  }

  // Get resolved type from MLIR (after type inference, before lowering)
  if (lastIsExpression) {
    resultType = codegenCtx.getResolvedReturnType();
  }

  if (!codegenCtx.lowerToStandard()) {
    std::cerr << "Lowering to standard failed: " << codegenCtx.getError()
              << "\n";
    return EvalResult::error("Code generation failed");
  }

  if (!codegenCtx.lowerToLLVM()) {
    std::cerr << "Lowering to LLVM failed: " << codegenCtx.getError() << "\n";
    return EvalResult::error("Code generation failed");
  }

  // Execute with JIT
  int64_t result = 0;
  if (!codegenCtx.runCode(result)) {
    std::cerr << "JIT execution failed: " << codegenCtx.getError() << "\n";
    return EvalResult::error("Execution failed");
  }

  // Update accumulated code on success
  accumulatedCode = fullCode + "\n";

  // Only return a value if the last statement was an expression
  if (lastIsExpression) {
    return EvalResult::value(result, resultType);
  }
  return EvalResult::ok();
}
