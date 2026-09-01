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

// ============== Tuple Lowering Tests ==============

TEST_F(ConversionPassTest, UnitTupleFunctionLowering) {
  OpBuilder builder(&context);
  auto unitType = polang::TupleType::get(&context, {});

  // Function taking and returning unit tuple
  auto funcType = builder.getFunctionType({unitType}, {unitType});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{"arg"});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Construct a unit tuple and return it
  auto unitVal =
      builder.create<polang::TupleOp>(builder.getUnknownLoc(), ValueRange{});
  builder.create<ReturnOp>(builder.getUnknownLoc(), unitVal.getResult());

  EXPECT_TRUE(runPass(module));

  // Verify polang ops are all lowered
  EXPECT_FALSE(hasOp<polang::FuncOp>(module));
  EXPECT_FALSE(hasOp<polang::TupleOp>(module));
  EXPECT_FALSE(hasOp<polang::ReturnOp>(module));

  // Verify func::FuncOp was created with 0 inputs and 0 outputs
  auto loweredFunc = module.lookupSymbol<func::FuncOp>("__polang_entry");
  ASSERT_TRUE(loweredFunc != nullptr);
  EXPECT_EQ(loweredFunc.getFunctionType().getNumInputs(), 0u);
  EXPECT_EQ(loweredFunc.getFunctionType().getNumResults(), 0u);

  // Verify no memref.alloca was created for unit tuple
  EXPECT_FALSE(hasOp<memref::AllocaOp>(module));

  // Verify func.return has 0 operands
  auto returnOp = cast<func::ReturnOp>(loweredFunc.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getNumOperands(), 0u);

  module->erase();
}

TEST_F(ConversionPassTest, TupleOpAndGetOpLowering) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);

  auto funcType = builder.getFunctionType({}, {i64Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto c1 = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 42));
  auto c2 = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 3.14));

  auto tuple = builder.create<TupleOp>(
      builder.getUnknownLoc(), ValueRange{c1.getResult(), c2.getResult()});
  auto get = builder.create<TupleGetOp>(builder.getUnknownLoc(),
                                        tuple.getResult(), 0);
  builder.create<ReturnOp>(builder.getUnknownLoc(), get.getResult());

  EXPECT_TRUE(runPass(module));

  EXPECT_FALSE(hasOp<polang::TupleOp>(module));
  EXPECT_FALSE(hasOp<polang::TupleGetOp>(module));
  EXPECT_TRUE(hasOp<memref::AllocaOp>(module));
  EXPECT_TRUE(hasOp<memref::StoreOp>(module));
  EXPECT_TRUE(hasOp<memref::LoadOp>(module));

  module->erase();
}

TEST_F(ConversionPassTest, TupleFunctionSretRoundTrip) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto tupleType = polang::TupleType::get(&context, {i64Type, f64Type});

  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // 1. Callee: make_pair(a: i64, b: f64) -> (i64, f64)
  auto calleeType = builder.getFunctionType({i64Type, f64Type}, {tupleType});
  auto callee = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "make_pair", calleeType,
      ArrayRef<StringRef>{"a", "b"});
  Block* calleeEntry = callee.addEntryBlock();
  builder.setInsertionPointToEnd(calleeEntry);
  auto t = builder.create<TupleOp>(
      builder.getUnknownLoc(),
      ValueRange{calleeEntry->getArgument(0), calleeEntry->getArgument(1)});
  builder.create<ReturnOp>(builder.getUnknownLoc(), t.getResult());

  // 2. Caller: caller() -> i64
  builder.setInsertionPointToEnd(module.getBody());
  auto callerType = builder.getFunctionType({}, {i64Type});
  auto caller = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "caller", callerType, ArrayRef<StringRef>{});
  Block* callerEntry = caller.addEntryBlock();
  builder.setInsertionPointToEnd(callerEntry);
  auto c1 = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 10));
  auto c2 = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 20.0));
  auto call = builder.create<CallOp>(
      builder.getUnknownLoc(), "make_pair", TypeRange{tupleType},
      ValueRange{c1.getResult(), c2.getResult()});
  auto get = builder.create<TupleGetOp>(builder.getUnknownLoc(),
                                        call.getResult(), 0);
  builder.create<ReturnOp>(builder.getUnknownLoc(), get.getResult());

  EXPECT_TRUE(runPass(module));

  // Check lowered make_pair: (memref<2xi64>, i64, f64) -> ()
  auto loweredCallee = module.lookupSymbol<func::FuncOp>("make_pair");
  ASSERT_TRUE(loweredCallee != nullptr);
  EXPECT_EQ(loweredCallee.getFunctionType().getNumInputs(), 3u);
  EXPECT_TRUE(isa<MemRefType>(loweredCallee.getFunctionType().getInput(0)));
  EXPECT_EQ(loweredCallee.getFunctionType().getNumResults(), 0u);

  // Check lowered caller: () -> i64
  auto loweredCaller = module.lookupSymbol<func::FuncOp>("caller");
  ASSERT_TRUE(loweredCaller != nullptr);
  EXPECT_EQ(loweredCaller.getFunctionType().getNumInputs(), 0u);
  EXPECT_EQ(loweredCaller.getFunctionType().getNumResults(), 1u);

  module->erase();
}

} // namespace
