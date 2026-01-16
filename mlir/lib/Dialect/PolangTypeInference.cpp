//===- PolangTypeInference.cpp - Type inference pass ------------*- C++ -*-===//
//
// This file implements the Polang type inference pass.
//
//===----------------------------------------------------------------------===//

#include "polang/Dialect/Passes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace polang;

namespace {

struct PolangTypeInferencePass
    : public PassWrapper<PolangTypeInferencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolangTypeInferencePass)

  StringRef getArgument() const final { return "polang-type-inference"; }
  StringRef getDescription() const final {
    return "Perform Polang-specific type inference";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Infer return types for functions without explicit return type
    module.walk([&](FuncOp funcOp) {
      if (funcOp.getResultTypes().empty()) {
        // Find return ops and infer type from returned value
        Type inferredType;
        funcOp.walk([&](ReturnOp returnOp) {
          if (returnOp.getValue())
            inferredType = returnOp.getValue().getType();
        });

        if (inferredType) {
          // Update function signature with inferred return type
          auto newType = FunctionType::get(
              funcOp.getContext(), funcOp.getArgumentTypes(), {inferredType});
          funcOp.setFunctionType(newType);
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> polang::createPolangTypeInferencePass() {
  return std::make_unique<PolangTypeInferencePass>();
}

void polang::registerPolangDialectPasses() {
  PassRegistration<PolangTypeInferencePass>();
}
