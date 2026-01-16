//===- mlir_codegen.cpp - MLIR-based code generation ------------*- C++ -*-===//
//
// This file implements the MLIR-based code generation context.
//
//===----------------------------------------------------------------------===//

#include "compiler/mlir_codegen.hpp"

#include "polang/Conversion/Passes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/MLIRGen.h"

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

using namespace mlir;

namespace polang {

MLIRCodeGenContext::MLIRCodeGenContext() = default;
MLIRCodeGenContext::~MLIRCodeGenContext() = default;

bool MLIRCodeGenContext::initializeContext() {
  if (context_)
    return true;

  context_ = std::make_unique<MLIRContext>();

  // Register all required dialects
  context_->getOrLoadDialect<PolangDialect>();
  context_->getOrLoadDialect<arith::ArithDialect>();
  context_->getOrLoadDialect<func::FuncDialect>();
  context_->getOrLoadDialect<scf::SCFDialect>();
  context_->getOrLoadDialect<cf::ControlFlowDialect>();
  context_->getOrLoadDialect<memref::MemRefDialect>();
  context_->getOrLoadDialect<LLVM::LLVMDialect>();

  // Register LLVM IR translation
  registerBuiltinDialectTranslation(*context_);
  registerLLVMDialectTranslation(*context_);

  return true;
}

bool MLIRCodeGenContext::generateCode(const NBlock& ast) {
  if (!initializeContext()) {
    error_ = "Failed to initialize MLIR context";
    return false;
  }

  auto moduleRef = mlirGen(*context_, ast);
  if (!moduleRef) {
    error_ = "Failed to generate MLIR from AST";
    return false;
  }

  module_ = std::make_unique<OwningOpRef<ModuleOp>>(std::move(moduleRef));
  return true;
}

bool MLIRCodeGenContext::lowerToStandard() {
  if (!module_ || !*module_) {
    error_ = "No module to lower";
    return false;
  }

  PassManager pm(context_.get());

  // Add the Polang to Standard lowering pass
  pm.addPass(createPolangToStandardPass());

  // Run canonicalization
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(**module_))) {
    error_ = "Failed to lower Polang dialect to standard dialects";
    return false;
  }

  return true;
}

bool MLIRCodeGenContext::lowerToLLVM() {
  if (!module_ || !*module_) {
    error_ = "No module to lower";
    return false;
  }

  PassManager pm(context_.get());

  // Lower SCF to CF
  pm.addPass(createConvertSCFToCFPass());

  // Lower to LLVM
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  // Reconcile unrealized casts
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(**module_))) {
    error_ = "Failed to lower to LLVM dialect";
    return false;
  }

  return true;
}

void MLIRCodeGenContext::printMLIR(llvm::raw_ostream& os) {
  if (module_ && *module_) {
    (*module_)->print(os);
  }
}

bool MLIRCodeGenContext::printLLVMIR(llvm::raw_ostream& os) {
  if (!module_ || !*module_) {
    error_ = "No module to translate";
    return false;
  }

  // Initialize LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create LLVM context for translation
  llvm::LLVMContext llvmContext;

  // Translate to LLVM IR
  auto llvmModule = translateModuleToLLVMIR(**module_, llvmContext);
  if (!llvmModule) {
    error_ = "Failed to translate MLIR to LLVM IR";
    return false;
  }

  llvmModule->print(os, nullptr);
  return true;
}

bool MLIRCodeGenContext::runCode(int64_t& result) {
  if (!module_ || !*module_) {
    error_ = "No module to execute";
    return false;
  }

  // Initialize LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create execution engine
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  ExecutionEngineOptions options;
  options.transformer = optPipeline;

  auto maybeEngine = ExecutionEngine::create(**module_, options);
  if (!maybeEngine) {
    error_ = "Failed to create execution engine: " +
             llvm::toString(maybeEngine.takeError());
    return false;
  }

  auto& engine = *maybeEngine;

  // Invoke main function using invokePacked
  // Note: The result is returned via a pointer argument
  int64_t resultVal = 0;
  llvm::SmallVector<void*> args;
  args.push_back(&resultVal);

  auto invocationResult = engine->invokePacked("main", args);
  if (invocationResult) {
    error_ = "Execution failed: " + llvm::toString(std::move(invocationResult));
    return false;
  }

  result = resultVal;
  return true;
}

} // namespace polang
