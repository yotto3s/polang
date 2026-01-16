#include "repl/repl_session.hpp"
#include "repl/input_checker.hpp"

#include "compiler/codegen.hpp"
#include "compiler/codegen_visitor.hpp"
#include "parser/node.hpp"
#include "parser/parser_api.hpp"
#include "parser/type_checker.hpp"

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>

#include <cstring>
#include <iostream>

using namespace llvm;

ReplSession::ReplSession() = default;
ReplSession::~ReplSession() = default;

bool ReplSession::initialize() {
  if (initialized_) {
    return true;
  }

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  initialized_ = true;
  return true;
}

bool ReplSession::isInputIncomplete(const std::string& input) {
  return InputChecker::isInputIncomplete(input);
}

EvalResult ReplSession::evaluate(const std::string& input) {
  if (!initialized_) {
    return EvalResult::error("REPL session not initialized");
  }

  // Combine accumulated code with new input
  const std::string fullCode = accumulatedCode_ + input;

  // Parse the combined input
  NBlock* ast = polang_parse(fullCode);
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
    const NStatement* lastStmt = ast->statements.back();
    // NExpressionStatement wraps expressions as statements
    if (dynamic_cast<const NExpressionStatement*>(lastStmt) != nullptr) {
      lastIsExpression = true;
      // Get the inferred type of the last expression
      TypeChecker checker;
      checker.check(*ast);
      resultType = checker.getInferredType();
    }
  }

  // Generate code
  CodeGenContext codegenCtx;
  codegenCtx.generateCode(*ast);

  // Verify the module
  std::string errStr;
  raw_string_ostream errStream(errStr);
  if (verifyModule(*codegenCtx.module, &errStream)) {
    std::cerr << "Module verification failed: " << errStr << "\n";
    return EvalResult::error("Code generation failed");
  }

  // Execute with JIT
  auto JTMB = orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB) {
    return EvalResult::error("Failed to detect host");
  }
  JTMB->setCPU("generic");

  auto JIT =
      orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(*JTMB)).create();
  if (!JIT) {
    return EvalResult::error("Failed to create JIT");
  }

  // Reparse and regenerate in a new context for JIT
  std::string irStr;
  raw_string_ostream irStream(irStr);
  codegenCtx.printIR(irStream);
  irStream.flush();

  auto jitContext = std::make_unique<LLVMContext>();
  SMDiagnostic parseErr;
  std::unique_ptr<Module> jitModule =
      parseAssemblyString(irStr, parseErr, *jitContext);
  if (!jitModule) {
    return EvalResult::error("Failed to parse IR for JIT");
  }

  auto TSM = orc::ThreadSafeModule(std::move(jitModule), std::move(jitContext));

  if (auto Err = (*JIT)->addIRModule(std::move(TSM))) {
    return EvalResult::error("Failed to add module to JIT");
  }

  auto MainSym = (*JIT)->lookup("main");
  if (!MainSym) {
    return EvalResult::error("Failed to find main function");
  }

  auto* MainFn = MainSym->toPtr<int64_t (*)()>();
  const int64_t result = MainFn();

  // Update accumulated code on success
  accumulatedCode_ = fullCode + "\n";

  // Only return a value if the last statement was an expression
  if (lastIsExpression) {
    return EvalResult::value(result, resultType);
  }
  return EvalResult::ok();
}
