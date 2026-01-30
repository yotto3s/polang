// Suppress warnings from LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "repl/repl_session.hpp"
#include "repl/input_checker.hpp"

#include "compiler/jit_session.hpp"
#include "compiler/mlir_codegen.hpp"
#include "parser/node.hpp"
#include "parser/parser_api.hpp"
#include "parser/type_checker.hpp"

#include "mlir/IR/BuiltinOps.h"

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

  // Initialize persistent JIT session
  jitSession = std::make_unique<polang::JITSession>();
  std::string jitError;
  if (!jitSession->initialize(jitError)) {
    std::cerr << "JIT initialization failed: " << jitError << "\n";
    return false;
  }

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

  // Initialize persistent TypeChecker on first evaluation
  if (!typeChecker) {
    typeChecker = std::make_unique<TypeChecker>();
  }

  // Snapshot TypeChecker state for rollback on error
  const auto snapshot = typeChecker->saveState();

  // Incrementally type-check only the new statements
  const bool isFirstEval = (previousStatementCount == 0);
  const auto errors = isFirstEval ? typeChecker->check(*newAst)
                                  : typeChecker->checkIncremental(*newAst);
  if (!errors.empty()) {
    // Rollback TypeChecker state on error
    typeChecker->restoreState(snapshot);
    std::string errMsg;
    for (const auto& err : errors) {
      errMsg += err.message;
      if (&err != &errors.back()) {
        errMsg += "\n";
      }
    }
    return EvalResult::error(errMsg);
  }

  // Merge new statements into accumulated AST (still needed for MLIRGen)
  if (!accumulatedAst) {
    accumulatedAst = std::make_unique<NBlock>();
  }
  for (auto& stmt : newAst->statements) {
    accumulatedAst->statements.push_back(std::move(stmt));
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

  // Generate unique entry function name for this evaluation
  const std::string entryFuncName =
      "__polang_eval_" + std::to_string(evalCounter);

  // Generate code using MLIR backend - always emit type variables.
  // Skip internal type checking since the persistent TypeChecker already ran.
  polang::MLIRCodeGenContext codegenCtx;
  const std::string inferredType = typeChecker->getInferredType();

  if (!codegenCtx.generateCode(*accumulatedAst, /*emitTypeVars=*/true,
                               /*skipTypeCheck=*/true, inferredType,
                               entryFuncName)) {
    std::cerr << "MLIR generation failed: " << codegenCtx.getError() << "\n";
    // Rollback on failure
    accumulatedAst->statements.resize(previousStatementCount);
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Code generation failed");
  }

  // Run type inference to resolve type variables
  if (!codegenCtx.runTypeInference()) {
    std::cerr << "Type inference failed: " << codegenCtx.getError() << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Type inference failed");
  }

  // Get resolved type from MLIR (after type inference, before lowering)
  if (lastIsExpression) {
    resultType = codegenCtx.getResolvedReturnType(entryFuncName);
  }

  if (!codegenCtx.lowerToStandard()) {
    std::cerr << "Lowering to standard failed: " << codegenCtx.getError()
              << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Code generation failed");
  }

  if (!codegenCtx.lowerToLLVM()) {
    std::cerr << "Lowering to LLVM failed: " << codegenCtx.getError() << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Code generation failed");
  }

  // Add compiled module to persistent JIT and execute
  auto mlirModule = codegenCtx.takeModule();
  std::string jitError;
  if (!jitSession->addModule(mlirModule, jitError)) {
    std::cerr << "JIT module addition failed: " << jitError << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    typeChecker->restoreState(snapshot);
    return EvalResult::error("JIT compilation failed");
  }

  int64_t result = 0;
  if (!jitSession->execute(entryFuncName, result, jitError, resultType)) {
    std::cerr << "JIT execution failed: " << jitError << "\n";
    accumulatedAst->statements.resize(previousStatementCount);
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Execution failed");
  }

  // Success — increment eval counter
  ++evalCounter;

  // Only return a value if the last statement was an expression
  if (lastIsExpression) {
    return EvalResult::value(result, resultType);
  }
  return EvalResult::ok();
}
