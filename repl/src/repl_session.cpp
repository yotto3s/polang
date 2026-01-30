// Suppress warnings from LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "repl/repl_session.hpp"
#include "repl/input_checker.hpp"

#include "compiler/compiled_symbols.hpp"
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

  // Parse the new input
  auto newAst = polang_parse(input);
  if (newAst == nullptr) {
    return EvalResult::error("Parse error");
  }

  // Initialize persistent TypeChecker on first evaluation
  if (!typeChecker) {
    typeChecker = std::make_unique<TypeChecker>();
  }

  // Snapshot TypeChecker state for rollback on error
  const auto snapshot = typeChecker->saveState();

  // Incrementally type-check only the new statements
  const bool isFirstEval = (evalCounter == 0);
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

  // Check if the last statement is an expression (should print result)
  bool lastIsExpression = false;
  std::string resultType = "void";
  if (!newAst->statements.empty()) {
    const NStatement* lastStmt = newAst->statements.back().get();
    if (dynamic_cast<const NExpressionStatement*>(lastStmt) != nullptr) {
      lastIsExpression = true;
    }
  }

  // Generate unique entry function name for this evaluation
  const std::string entryFuncName =
      "__polang_eval_" + std::to_string(evalCounter);

  // Generate code using MLIR backend from only the new AST.
  // Pass compiledSymbols so MLIRGen can emit extern declarations
  // for previously compiled symbols (incremental mode).
  polang::MLIRCodeGenContext codegenCtx;
  const std::string inferredType = typeChecker->getInferredType();

  // Initialize compiledSymbols on first eval
  if (!compiledSymbols) {
    compiledSymbols = std::make_unique<polang::CompiledSymbols>();
  }

  // Always pass compiledSymbols so variables are emitted as globals
  // (empty on first eval, populated on subsequent evals)
  polang::OptCompiledSymbols optSymbols = std::cref(*compiledSymbols);

  if (!codegenCtx.generateCode(*newAst, /*emitTypeVars=*/true,
                               /*skipTypeCheck=*/true, inferredType,
                               entryFuncName, optSymbols)) {
    std::cerr << "MLIR generation failed: " << codegenCtx.getError() << "\n";
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Code generation failed");
  }

  // Run type inference to resolve type variables
  if (!codegenCtx.runTypeInference()) {
    std::cerr << "Type inference failed: " << codegenCtx.getError() << "\n";
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
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Code generation failed");
  }

  if (!codegenCtx.lowerToLLVM()) {
    std::cerr << "Lowering to LLVM failed: " << codegenCtx.getError() << "\n";
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Code generation failed");
  }

  // Add compiled module to persistent JIT and execute
  auto mlirModule = codegenCtx.takeModule();
  std::string jitError;
  if (!jitSession->addModule(mlirModule, jitError)) {
    std::cerr << "JIT module addition failed: " << jitError << "\n";
    typeChecker->restoreState(snapshot);
    return EvalResult::error("JIT compilation failed");
  }

  int64_t result = 0;
  if (!jitSession->execute(entryFuncName, result, jitError, resultType)) {
    std::cerr << "JIT execution failed: " << jitError << "\n";
    typeChecker->restoreState(snapshot);
    return EvalResult::error("Execution failed");
  }

  // Success — update compiled symbols and increment eval counter
  updateCompiledSymbols(*newAst);
  ++evalCounter;

  // Only return a value if the last statement was an expression
  if (lastIsExpression) {
    return EvalResult::value(result, resultType);
  }
  return EvalResult::ok();
}

void ReplSession::updateCompiledSymbols(NBlock& newAst) {
  // Use TypeChecker's snapshot to get fully resolved types
  const auto currentState = typeChecker->saveState();

  // Helper to register a function declaration with optional name prefix
  auto registerFunc = [&](NFunctionDeclaration* funcDecl,
                          const std::string& prefix,
                          std::unique_ptr<NStatement>& stmt) {
    const std::string mangledName = prefix + funcDecl->id->name;
    polang::CompiledFunction func;
    func.name = mangledName;

    // Get return type from TypeChecker
    auto retIt = currentState.functionReturnTypes.find(mangledName);
    if (retIt != currentState.functionReturnTypes.end()) {
      func.returnType = retIt->second;
    } else if (funcDecl->type) {
      func.returnType = funcDecl->type->getTypeName();
    } else {
      func.returnType = "i64";
    }

    // Get param types from TypeChecker
    auto paramIt = currentState.functionParamTypes.find(mangledName);
    if (paramIt != currentState.functionParamTypes.end()) {
      func.paramTypes = paramIt->second;
    } else {
      for (const auto& arg : funcDecl->arguments) {
        func.paramTypes.push_back(arg->type ? arg->type->getTypeName() : "i64");
      }
    }

    for (const auto& arg : funcDecl->arguments) {
      func.paramNames.push_back(arg->id->name);
    }
    for (const auto& capture : funcDecl->captures) {
      func.captureNames.push_back(capture.id->name);
      func.captureTypes.push_back(capture.type ? capture.type->getTypeName()
                                               : "i64");
    }

    func.isGeneric = !funcDecl->typeParams.empty();
    func.typeParams = funcDecl->typeParams;
    func.typeParamBounds = funcDecl->typeParamBounds;

    compiledSymbols->functions[func.name] = func;

    if (func.isGeneric) {
      compiledSymbols->genericFuncAstNodes.push_back(std::move(stmt));
    }
  };

  for (auto& stmt : newAst.statements) {
    // Variable declarations become globals
    if (auto* varDecl = dynamic_cast<NVariableDeclaration*>(stmt.get())) {
      polang::CompiledGlobal global;
      global.name = varDecl->id->name;

      auto typeIt = currentState.localTypes.find(varDecl->id->name);
      if (typeIt != currentState.localTypes.end()) {
        global.type = typeIt->second;
      } else if (varDecl->type) {
        global.type = varDecl->type->getTypeName();
      } else {
        global.type = "i64";
      }

      compiledSymbols->globals[global.name] = global;
      continue;
    }

    // Function declarations
    if (auto* funcDecl = dynamic_cast<NFunctionDeclaration*>(stmt.get())) {
      registerFunc(funcDecl, "", stmt);
      continue;
    }

    // Module declarations - register members with mangled names
    if (auto* modDecl = dynamic_cast<NModuleDeclaration*>(stmt.get())) {
      const std::string prefix = modDecl->name->name + "$$";
      for (auto& member : modDecl->members) {
        if (auto* funcDecl =
                dynamic_cast<NFunctionDeclaration*>(member.get())) {
          registerFunc(funcDecl, prefix, member);
        } else if (auto* varDecl =
                       dynamic_cast<NVariableDeclaration*>(member.get())) {
          const std::string mangledName = prefix + varDecl->id->name;
          polang::CompiledGlobal global;
          global.name = mangledName;
          auto typeIt = currentState.localTypes.find(mangledName);
          if (typeIt != currentState.localTypes.end()) {
            global.type = typeIt->second;
          } else if (varDecl->type) {
            global.type = varDecl->type->getTypeName();
          } else {
            global.type = "i64";
          }
          compiledSymbols->globals[global.name] = global;
        }
      }
      continue;
    }

    // Import statements - track import mappings for name resolution
    if (auto* importStmt = dynamic_cast<NImportStatement*>(stmt.get())) {
      const std::string moduleName = importStmt->modulePath->getMangledName();

      switch (importStmt->kind) {
      case ImportKind::Items:
        for (const auto& item : importStmt->items) {
          const std::string mangled = moduleName + "$$" + item.name;
          const std::string localName = item.getEffectiveName();
          compiledSymbols->importMappings[localName] = mangled;
        }
        break;
      case ImportKind::All: {
        const std::string prefix = moduleName + "$$";
        for (const auto& [mangled, func] : compiledSymbols->functions) {
          if (mangled.substr(0, prefix.size()) == prefix) {
            const std::string localName = mangled.substr(prefix.size());
            if (localName.find("$$") == std::string::npos) {
              compiledSymbols->importMappings[localName] = mangled;
            }
          }
        }
        for (const auto& [mangled, global] : compiledSymbols->globals) {
          if (mangled.substr(0, prefix.size()) == prefix) {
            const std::string localName = mangled.substr(prefix.size());
            if (localName.find("$$") == std::string::npos) {
              compiledSymbols->importMappings[localName] = mangled;
            }
          }
        }
        break;
      }
      default:
        break;
      }
    }
  }
}
