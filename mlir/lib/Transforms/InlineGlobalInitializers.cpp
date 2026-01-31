//===- InlineGlobalInitializers.cpp - Inline global init regions -*- C++
//-*-===//
//
// This pass inlines GlobalOp initializer regions into the entry function.
// It must run before monomorphization so that ops inside init regions
// (e.g., LetExprOp bindings) are visible in the normal function scope
// and don't cause symbol conflicts during type inference.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#pragma GCC diagnostic pop

using namespace mlir;
using namespace polang;

namespace {

struct InlineGlobalInitializersPass
    : public PassWrapper<InlineGlobalInitializersPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineGlobalInitializersPass)

  [[nodiscard]] StringRef getArgument() const final {
    return "polang-inline-global-initializers";
  }
  [[nodiscard]] StringRef getDescription() const final {
    return "Inline GlobalOp initializer regions into the entry function";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<PolangDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Find the last polang::FuncOp with a body — that's the entry function
    FuncOp entryFunc;
    for (auto func : moduleOp.getOps<FuncOp>()) {
      if (!func.getBody().empty()) {
        entryFunc = func;
      }
    }
    if (!entryFunc) {
      return;
    }

    auto& entryBlock = entryFunc.getBody().front();
    OpBuilder builder(&getContext());

    for (auto globalOp :
         llvm::make_early_inc_range(moduleOp.getOps<GlobalOp>())) {
      if (globalOp.getInitializer().empty()) {
        continue;
      }

      auto& initBlock = globalOp.getInitializer().front();
      auto yieldOp = cast<YieldGlobalOp>(initBlock.getTerminator());
      Value yieldedValue = yieldOp.getValue();

      // Replace YieldGlobalOp with GlobalStoreOp (still in the init block)
      builder.setInsertionPoint(yieldOp);
      builder.create<GlobalStoreOp>(yieldOp.getLoc(), globalOp.getSymName(),
                                    yieldedValue);
      yieldOp.erase();

      // Move all ops from init block into the start of the entry function
      entryBlock.getOperations().splice(entryBlock.begin(),
                                        initBlock.getOperations());

      // Clear the now-empty region so the GlobalOp verifier doesn't
      // complain about an empty initializer block.
      globalOp.getInitializer().getBlocks().clear();
    }
  }
};

} // namespace

std::unique_ptr<Pass> polang::createInlineGlobalInitializersPass() {
  return std::make_unique<InlineGlobalInitializersPass>();
}
