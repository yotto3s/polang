//===- conversion_pass_test.cpp - Test PolangToStandard pass -----*- C++ -*-===//
//
// Tests that exercise code paths in PolangToStandard.cpp by constructing
// MLIR with AllocaOp and PrintOp programmatically.
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

// ============== AllocaOp Lowering ==============

TEST_F(ConversionPassTest, AllocaOpLowering) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto memRefType = MemRefType::get({}, builder.getIntegerType(64));

  auto funcType = builder.getFunctionType({}, {i64Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Create an AllocaOp
  builder.create<AllocaOp>(
      builder.getUnknownLoc(), memRefType,
      "x", (Type)i64Type,
      /*isMutable=*/false);

  // Return a constant (so the function is complete)
  auto retVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 42));
  builder.create<ReturnOp>(builder.getUnknownLoc(), retVal.getResult());

  EXPECT_TRUE(runPass(module));

  // Verify AllocaOp was lowered (no polang.alloca remains)
  EXPECT_FALSE(hasOp<polang::AllocaOp>(module));
  // Verify memref.alloca was created
  EXPECT_TRUE(hasOp<memref::AllocaOp>(module));

  module->erase();
}

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

// ============== AllocaOp with Mutable Flag ==============

TEST_F(ConversionPassTest, AllocaOpMutable) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto memRefType = MemRefType::get({}, builder.getIntegerType(64));

  auto funcType = builder.getFunctionType({}, {i64Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Create a mutable AllocaOp
  builder.create<AllocaOp>(
      builder.getUnknownLoc(), memRefType,
      "y", (Type)i64Type,
      /*isMutable=*/true);

  auto retVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 99));
  builder.create<ReturnOp>(builder.getUnknownLoc(), retVal.getResult());

  EXPECT_TRUE(runPass(module));

  // Verify the mutable AllocaOp was also lowered
  EXPECT_FALSE(hasOp<polang::AllocaOp>(module));
  EXPECT_TRUE(hasOp<memref::AllocaOp>(module));

  module->erase();
}

// ============== AllocaOp with Float Type ==============

TEST_F(ConversionPassTest, AllocaOpFloatType) {
  OpBuilder builder(&context);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto memRefType = MemRefType::get({}, builder.getF64Type());

  auto funcType = builder.getFunctionType({}, {f64Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Create an AllocaOp for a float variable
  builder.create<AllocaOp>(
      builder.getUnknownLoc(), memRefType,
      "z", (Type)f64Type,
      /*isMutable=*/false);

  auto retVal = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 3.14));
  builder.create<ReturnOp>(builder.getUnknownLoc(), retVal.getResult());

  EXPECT_TRUE(runPass(module));

  // Verify polang.alloca was lowered to memref.alloca
  EXPECT_FALSE(hasOp<polang::AllocaOp>(module));
  EXPECT_TRUE(hasOp<memref::AllocaOp>(module));

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
