//===- polang-opt.cpp - Polang MLIR optimizer driver ----------*- C++ -*-===//
//
// A minimal mlir-opt wrapper that registers Polang dialects.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "polang/Conversion/Passes.h"
#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Transforms/Passes.h"

#pragma GCC diagnostic pop

namespace {

/// Register all Polang passes
void registerPolangPasses() {
  // Register name resolution pass
  mlir::PassPipelineRegistration<>(
      "polang-resolve-names", "Resolve variable references to SSA values",
      [](mlir::OpPassManager& pm) {
        pm.addPass(polang::createNameResolutionPass());
      });

  // Register type check pass
  mlir::PassPipelineRegistration<>(
      "polang-type-check", "Validate type correctness of AST dialect IR",
      [](mlir::OpPassManager& pm) {
        pm.addPass(polang::createTypeCheckPass());
      });

  // Register type inference pass
  mlir::PassPipelineRegistration<>(
      "polang-type-inference", "Infer types for type variables",
      [](mlir::OpPassManager& pm) {
        pm.addPass(polang::createTypeInferencePass());
      });

  // Register monomorphization pass
  mlir::PassPipelineRegistration<>(
      "polang-monomorphize", "Monomorphize polymorphic functions",
      [](mlir::OpPassManager& pm) {
        pm.addPass(polang::createMonomorphizationPass());
      });

  // Register AST-to-Polang conversion pass
  mlir::PassPipelineRegistration<>(
      "polang-ast-to-polang", "Convert AST dialect to Polang dialect",
      [](mlir::OpPassManager& pm) {
        pm.addPass(polang::createASTToPolangPass());
      });

  // Register Polang-to-Standard lowering pass
  mlir::PassPipelineRegistration<>(
      "convert-polang-to-standard", "Lower Polang dialect to standard dialects",
      [](mlir::OpPassManager& pm) {
        pm.addPass(polang::createPolangToStandardPass());
      });
}

} // namespace

int main(int argc, char** argv) {
  registerPolangPasses();

  mlir::DialectRegistry registry;
  registry.insert<polang::PolangDialect>();
  registry.insert<polang::ast::PolangASTDialect>();
  // Register dialects needed for lowering
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Polang optimizer", registry));
}
