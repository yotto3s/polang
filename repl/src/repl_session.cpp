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

  // Parse the new input separately
  auto newAst = polang_parse(input);
  if (newAst == nullptr) {
    return EvalResult::error("Parse error");
  }

  // Count statements before merging (to identify new statements)
  const size_t previousStatementCount =
      accumulatedAst ? accumulatedAst->statements.size() : 0;

  // Merge new statements into accumulated AST
  if (!accumulatedAst) {
    accumulatedAst = std::make_unique<NBlock>();
  }
  for (auto& stmt : newAst->statements) {
    accumulatedAst->statements.push_back(std::move(stmt));
  }

  // Type check the combined AST
  const auto errors = polang_check_types(*accumulatedAst);
  if (!errors.empty()) {
    // Rollback: remove newly added statements on error
    accumulatedAst->statements.resize(previousStatementCount);
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
  if (!accumulatedAst->statements.empty()) {
    const NStatement* lastStmt = accumulatedAst->statements.back().get();
    // NExpressionStatement wraps expressions as statements
    if (dynamic_cast<const NExpressionStatement*>(lastStmt) != nullptr) {
      lastIsExpression = true;
      // Type will be resolved from MLIR after type inference
    }
  }

  // Generate code using MLIR backend - always emit type variables
  polang::MLIRCodeGenContext codegenCtx;

  if (!codegenCtx.generateCode(*accumulatedAst, true)) {
    std::cerr << "MLIR generation failed: " << codegenCtx.getError() << "\n";
    // Rollback on failure
    accumulatedAst->statements.resize(previousStatementCount);
    return EvalResult::error("Code generation failed");
  }

  // Run type inference to resolve type variables
  if (!codegenCtx.runTypeInference()) {
    std::cerr << "Type inference failed: " << codegenCtx.getError() << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    return EvalResult::error("Type inference failed");
  }

  // Get resolved type from MLIR (after type inference, before lowering)
  if (lastIsExpression) {
    resultType = codegenCtx.getResolvedReturnType();
  }

  if (!codegenCtx.lowerToStandard()) {
    std::cerr << "Lowering to standard failed: " << codegenCtx.getError()
              << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    return EvalResult::error("Code generation failed");
  }

  if (!codegenCtx.lowerToLLVM()) {
    std::cerr << "Lowering to LLVM failed: " << codegenCtx.getError() << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    return EvalResult::error("Code generation failed");
  }

  // Execute with JIT
  int64_t result = 0;
  if (!codegenCtx.runCode(result)) {
    std::cerr << "JIT execution failed: " << codegenCtx.getError() << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    return EvalResult::error("Execution failed");
  }

  // Success - keep the merged AST (already merged above)

  // Only return a value if the last statement was an expression
  if (lastIsExpression) {
    return EvalResult::value(result, resultType);
  }
  return EvalResult::ok();
}
