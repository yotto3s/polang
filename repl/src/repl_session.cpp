#include "repl/repl_session.hpp"
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
  int parenDepth = 0;
  int letWithoutIn = 0;
  int ifWithoutElse = 0;

  // Simple tokenization - track keywords and brackets
  size_t i = 0;
  while (i < input.size()) {
    // Skip whitespace
    if (std::isspace(static_cast<unsigned char>(input[i]))) {
      ++i;
      continue;
    }

    // Track parentheses
    if (input[i] == '(') {
      ++parenDepth;
      ++i;
      continue;
    }
    if (input[i] == ')') {
      --parenDepth;
      ++i;
      continue;
    }

    // Check for keywords
    if (std::isalpha(static_cast<unsigned char>(input[i])) || input[i] == '_') {
      const size_t start = i;
      while (i < input.size() &&
             (std::isalnum(static_cast<unsigned char>(input[i])) ||
              input[i] == '_')) {
        ++i;
      }
      const std::string word = input.substr(start, i - start);

      if (word == "let") {
        // Check if this is followed by identifier then '(' (function) or '='
        // (variable) For let-expressions, they need 'in'
        // For simplicity, track potential let-expressions
        ++letWithoutIn;
      } else if (word == "in") {
        if (letWithoutIn > 0) {
          --letWithoutIn;
        }
      } else if (word == "if") {
        ++ifWithoutElse;
      } else if (word == "else") {
        if (ifWithoutElse > 0) {
          --ifWithoutElse;
        }
      }
      continue;
    }

    // Skip other characters
    ++i;
  }

  // Input is incomplete if we have unbalanced structures
  return parenDepth > 0 || letWithoutIn > 0 || ifWithoutElse > 0;
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

  // Generate code
  CodeGenContext codegenCtx;
  codegenCtx.generateCode(*ast);

  // Get the inferred type of the last expression
  TypeChecker checker;
  checker.check(*ast);
  const std::string resultType = checker.getInferredType();

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

  return EvalResult::value(result, resultType);
}
