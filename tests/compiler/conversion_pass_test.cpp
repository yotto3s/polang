//===- conversion_pass_test.cpp - Test PolangToStandard pass -----*- C++ -*-===//
//
// Tests that exercise code paths in PolangToStandard.cpp by constructing
// MLIR with PrintOp programmatically.
//
//===----------------------------------------------------------------------===//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Conversion/Passes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

#pragma GCC diagnostic pop

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace mlir;
using namespace polang;

namespace {

class ConversionPassTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.getOrLoadDialect<PolangDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    // Capture diagnostics as strings
    diagHandler = context.getDiagEngine().registerHandler(
        [this](Diagnostic& diag) {
          diagnostics.push_back(diag.str());
          return success();
        });
  }

  void TearDown() override {
    context.getDiagEngine().eraseHandler(diagHandler);
  }

  /// Run the PolangToStandard conversion pass on a module.
  /// Returns true if the pass succeeded, false if it failed.
  bool runPass(ModuleOp module) {
    PassManager pm(&context);
    pm.enableVerifier(false); // Disable verification for test flexibility
    pm.addPass(polang::createPolangToStandardPass());
    return succeeded(pm.run(module));
  }

  /// Check if any operation of the given type exists in the module.
  template <typename OpT>
  bool hasOp(ModuleOp module) {
    bool found = false;
    module.walk([&](OpT) { found = true; });
    return found;
  }

  MLIRContext context;
  std::vector<std::string> diagnostics;

private:
  DiagnosticEngine::HandlerID diagHandler{};
};

// ============== PrintOp Lowering ==============

TEST_F(ConversionPassTest, PrintOpLowering) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);

  auto funcType = builder.getFunctionType({}, {i64Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Create a value and print it
  auto val = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 42));
  builder.create<PrintOp>(builder.getUnknownLoc(), val.getResult());

  // Return a constant
  builder.create<ReturnOp>(builder.getUnknownLoc(), val.getResult());

  EXPECT_TRUE(runPass(module));

  // Verify PrintOp was erased (no polang.print remains)
  EXPECT_FALSE(hasOp<polang::PrintOp>(module));

  module->erase();
}

// ============== ConstantFloatOp with f32 Type ==============

TEST_F(ConversionPassTest, ConstantFloatOpF32) {
  OpBuilder builder(&context);
  auto f32Type = polang::FloatType::get(&context, 32);

  // Function returning f32
  auto funcType = builder.getFunctionType({}, {f32Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Create a ConstantFloatOp with f32 result type
  // Use f32 semantics for the APFloat value
  llvm::APFloat f32Val(2.5f);
  auto val = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f32Type,
      FloatAttr::get(builder.getF32Type(), f32Val));
  builder.create<ReturnOp>(builder.getUnknownLoc(), val.getResult());

  EXPECT_TRUE(runPass(module));

  // Verify ConstantFloatOp was lowered (no polang.constant_float remains)
  EXPECT_FALSE(hasOp<polang::ConstantFloatOp>(module));
  // Verify arith::ConstantOp was created
  EXPECT_TRUE(hasOp<arith::ConstantOp>(module));

  module->erase();
}

} // namespace
