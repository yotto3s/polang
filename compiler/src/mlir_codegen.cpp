//===- mlir_codegen.cpp - MLIR-based code generation ------------*- C++ -*-===//
//
// This file implements the MLIR-based code generation context.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "compiler/mlir_codegen.hpp"

#include "polang/Conversion/Passes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"
#include "polang/MLIRGen.h"
#include "polang/Transforms/Passes.h"

#include "parser/node.hpp"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#pragma GCC diagnostic pop

#include <mutex>

using namespace mlir;

namespace {

/// Ensure LLVM targets are initialized exactly once.
/// Thread-safe using std::call_once.
void ensureLLVMTargetsInitialized() {
  static std::once_flag llvmInitFlag;
  std::call_once(llvmInitFlag, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
}

} // namespace

namespace polang {

MLIRCodeGenContext::MLIRCodeGenContext() = default;
MLIRCodeGenContext::~MLIRCodeGenContext() = default;

bool MLIRCodeGenContext::initializeContext() {
  if (context) {
    return true;
  }

  context = std::make_unique<MLIRContext>();

  // Register all required dialects
  context->getOrLoadDialect<PolangDialect>();
  context->getOrLoadDialect<arith::ArithDialect>();
  context->getOrLoadDialect<func::FuncDialect>();
  context->getOrLoadDialect<scf::SCFDialect>();
  context->getOrLoadDialect<cf::ControlFlowDialect>();
  context->getOrLoadDialect<memref::MemRefDialect>();
  context->getOrLoadDialect<LLVM::LLVMDialect>();

  // Register LLVM IR translation
  registerBuiltinDialectTranslation(*context);
  registerLLVMDialectTranslation(*context);

  return true;
}

bool MLIRCodeGenContext::generateCode(const NBlock& ast, bool emitTypeVars) {
  if (!initializeContext()) {
    error = "Failed to initialize MLIR context";
    return false;
  }

  auto moduleRef = mlirGen(*context, ast, emitTypeVars);
  if (!moduleRef) {
    error = "Failed to generate MLIR from AST";
    return false;
  }

  module = std::make_unique<OwningOpRef<ModuleOp>>(std::move(moduleRef));
  return true;
}

bool MLIRCodeGenContext::runTypeInference() {
  if (!module || !*module) {
    error = "No module for type inference";
    return false;
  }

  PassManager pm(context.get());

  // Add the type inference pass - collects constraints and resolves types
  // for non-polymorphic functions. Polymorphic functions are preserved
  // with type variables.
  pm.addPass(polang::createTypeInferencePass());

  // Add the monomorphization pass - creates specialized versions of
  // polymorphic functions for each unique call signature.
  pm.addPass(polang::createMonomorphizationPass());

  // Note: We don't run canonicalization here to preserve all operations
  // for debugging/testing. Canonicalization happens in later lowering stages.

  if (failed(pm.run(**module))) {
    error = "Type inference failed";
    return false;
  }

  return true;
}

bool MLIRCodeGenContext::lowerToStandard() {
  if (!module || !*module) {
    error = "No module to lower";
    return false;
  }

  PassManager pm(context.get());

  // Add the Polang to Standard lowering pass
  pm.addPass(createPolangToStandardPass());

  // Run canonicalization
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(**module))) {
    error = "Failed to lower Polang dialect to standard dialects";
    return false;
  }

  return true;
}

bool MLIRCodeGenContext::lowerToLLVM() {
  if (!module || !*module) {
    error = "No module to lower";
    return false;
  }

  PassManager pm(context.get());

  // Lower SCF to CF
  pm.addPass(createConvertSCFToCFPass());

  // Lower to LLVM
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  // Reconcile unrealized casts
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(**module))) {
    error = "Failed to lower to LLVM dialect";
    return false;
  }

  return true;
}

void MLIRCodeGenContext::printMLIR(llvm::raw_ostream& os) {
  if (module && *module) {
    (*module)->print(os);
  }
}

bool MLIRCodeGenContext::printLLVMIR(llvm::raw_ostream& os) {
  if (!module || !*module) {
    error = "No module to translate";
    return false;
  }

  // Initialize LLVM targets (thread-safe, only executes once)
  ensureLLVMTargetsInitialized();

  // Create LLVM context for translation
  llvm::LLVMContext llvmContext;

  // Translate to LLVM IR
  auto llvmModule = translateModuleToLLVMIR(**module, llvmContext);
  if (!llvmModule) {
    error = "Failed to translate MLIR to LLVM IR";
    return false;
  }

  llvmModule->print(os, nullptr);
  return true;
}

bool MLIRCodeGenContext::runCode(int64_t& result) {
  if (!module || !*module) {
    error = "No module to execute";
    return false;
  }

  // Initialize LLVM targets (thread-safe, only executes once)
  ensureLLVMTargetsInitialized();

  // Create execution engine
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  ExecutionEngineOptions options;
  options.transformer = optPipeline;

  auto maybeEngine = ExecutionEngine::create(**module, options);
  if (!maybeEngine) {
    error = "Failed to create execution engine: " +
            llvm::toString(maybeEngine.takeError());
    return false;
  }

  auto& engine = *maybeEngine;

  // Invoke main function using invokePacked
  // Note: The result is returned via a pointer argument
  int64_t resultVal = 0;
  llvm::SmallVector<void*> args;
  args.push_back(&resultVal);

  auto invocationResult = engine->invokePacked("__polang_entry", args);
  if (invocationResult) {
    error = "Execution failed: " + llvm::toString(std::move(invocationResult));
    return false;
  }

  result = resultVal;
  return true;
}

std::string MLIRCodeGenContext::getResolvedReturnType() const {
  if (!module || !*module) {
    return "unknown";
  }

  // Find the __polang_entry function
  auto entryFunc = (*module)->lookupSymbol<polang::FuncOp>("__polang_entry");
  if (!entryFunc) {
    return "unknown";
  }

  FunctionType funcType = entryFunc.getFunctionType();
  if (funcType.getNumResults() == 0) {
    return "void";
  }

  Type returnType = funcType.getResult(0);
  if (auto intType = dyn_cast<polang::IntegerType>(returnType)) {
    std::string prefix = intType.isSigned() ? "i" : "u";
    return prefix + std::to_string(intType.getWidth());
  }
  if (auto floatType = dyn_cast<polang::FloatType>(returnType)) {
    return "f" + std::to_string(floatType.getWidth());
  }
  if (isa<polang::BoolType>(returnType)) {
    return "bool";
  }

  return "unknown";
}

} // namespace polang
