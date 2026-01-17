//===- Monomorphization.cpp - Function specialization -----------*- C++ -*-===//
//
// This file implements the monomorphization pass for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#include "polang/Transforms/Passes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace polang;

namespace {

struct MonomorphizationPass
    : public PassWrapper<MonomorphizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MonomorphizationPass)

  StringRef getArgument() const override { return "polang-monomorphize"; }
  StringRef getDescription() const override {
    return "Monomorphize polymorphic functions";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<PolangDialect>();
  }

  void runOnOperation() override {
    // TODO: Implement monomorphization
    // For now, this pass is a no-op since type inference resolves all types
    // to concrete types within a single module.
    // Monomorphization will be needed when we support separate compilation
    // or when functions are called with different type arguments.
  }
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createMonomorphizationPass() {
  return std::make_unique<MonomorphizationPass>();
}

} // namespace polang
