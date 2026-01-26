//===- polang-opt.cpp - Polang MLIR optimizer driver ----------*- C++ -*-===//
//
// A minimal mlir-opt wrapper that registers Polang dialects.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

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
}

} // namespace

int main(int argc, char** argv) {
  registerPolangPasses();

  mlir::DialectRegistry registry;
  registry.insert<polang::PolangDialect>();
  registry.insert<polang::ast::PolangASTDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Polang optimizer", registry));
}
