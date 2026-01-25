//===- type_inference_pass_test.cpp - Test TypeInference pass ----*- C++ -*-===//
//
// Tests that exercise error paths in TypeInference.cpp by constructing
// invalid MLIR programmatically (bypassing the parser/type checker).
//
//===----------------------------------------------------------------------===//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"
#include "polang/Transforms/Passes.h"

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

class TypeInferencePassTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.getOrLoadDialect<PolangDialect>();
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

  /// Run the TypeInference pass on a module.
  /// Returns true if the pass succeeded, false if it failed.
  bool runPass(ModuleOp module) {
    PassManager pm(&context);
    pm.enableVerifier(false); // Disable verification to test pass error paths
    pm.addPass(polang::createTypeInferencePass());
    return succeeded(pm.run(module));
  }

  /// Check if any diagnostic contains the given substring.
  bool hasDiagContaining(const std::string& substr) const {
    for (const auto& diag : diagnostics) {
      if (diag.find(substr) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  MLIRContext context;
  std::vector<std::string> diagnostics;

private:
  DiagnosticEngine::HandlerID diagHandler{};
};

// ============== Return Type Mismatch ==============

TEST_F(TypeInferencePassTest, ReturnTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);

  // Function declares return type i64
  auto funcType = builder.getFunctionType({}, {i64Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Return f64 value (mismatch with i64 return type)
  auto val = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 1.0));
  builder.create<ReturnOp>(builder.getUnknownLoc(), val.getResult());

  EXPECT_FALSE(runPass(module));
  EXPECT_TRUE(hasDiagContaining("return type mismatch"));

  module->erase();
}

// ============== Operand Type Mismatch ==============

TEST_F(TypeInferencePassTest, AddOpOperandTypeMismatch) {
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

  // AddOp with mismatched operand types: i64 + f64
  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 2.0));
  auto add = builder.create<AddOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());
  builder.create<ReturnOp>(builder.getUnknownLoc(), add.getResult());

  EXPECT_FALSE(runPass(module));
  EXPECT_TRUE(hasDiagContaining("operand type mismatch"));

  module->erase();
}

// ============== Result Type Mismatch ==============

TEST_F(TypeInferencePassTest, AddOpResultTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);

  auto funcType = builder.getFunctionType({}, {f64Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // AddOp with i64 operands but f64 result type
  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  auto rhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 2));
  auto add = builder.create<AddOp>(
      builder.getUnknownLoc(), f64Type, lhs.getResult(), rhs.getResult());
  builder.create<ReturnOp>(builder.getUnknownLoc(), add.getResult());

  EXPECT_FALSE(runPass(module));
  EXPECT_TRUE(hasDiagContaining("result type mismatch"));

  module->erase();
}

// ============== If Condition Not Bool ==============

TEST_F(TypeInferencePassTest, IfConditionNotBool) {
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

  // Use i64 as condition (should be bool)
  auto cond = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));

  auto ifOp = builder.create<polang::IfOp>(
      builder.getUnknownLoc(), i64Type, cond.getResult());

  // Add yield to then region
  builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
  auto thenVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 10));
  builder.create<YieldOp>(builder.getUnknownLoc(), thenVal.getResult());

  // Add yield to else region
  builder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
  auto elseVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 20));
  builder.create<YieldOp>(builder.getUnknownLoc(), elseVal.getResult());

  // Return the if result
  builder.setInsertionPointToEnd(entry);
  builder.create<ReturnOp>(builder.getUnknownLoc(), ifOp.getResult());

  EXPECT_FALSE(runPass(module));
  EXPECT_TRUE(hasDiagContaining("if condition must be bool"));

  module->erase();
}

// ============== Call Argument Type Mismatch ==============

TEST_F(TypeInferencePassTest, CallArgTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);

  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Define a target function that takes i64
  auto targetType = builder.getFunctionType({i64Type}, {i64Type});
  auto target = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "target", targetType, ArrayRef<StringRef>{});
  Block* targetEntry = target.addEntryBlock();
  builder.setInsertionPointToEnd(targetEntry);
  builder.create<ReturnOp>(builder.getUnknownLoc(),
                           targetEntry->getArgument(0));

  // Define caller function
  auto callerType = builder.getFunctionType({}, {i64Type});
  auto caller = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", callerType,
      ArrayRef<StringRef>{});
  Block* callerEntry = caller.addEntryBlock();
  builder.setInsertionPointToEnd(callerEntry);

  // Call target with f64 argument (mismatch with i64 parameter)
  auto arg = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 1.0));
  auto call = builder.create<CallOp>(
      builder.getUnknownLoc(), "target",
      TypeRange{i64Type}, ValueRange{arg.getResult()});
  builder.create<ReturnOp>(builder.getUnknownLoc(), call.getResult());

  EXPECT_FALSE(runPass(module));
  EXPECT_TRUE(hasDiagContaining("argument type mismatch"));

  module->erase();
}

// ============== Call Return Type Mismatch ==============

TEST_F(TypeInferencePassTest, CallReturnTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);

  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Define a target function that returns i64
  auto targetType = builder.getFunctionType({}, {i64Type});
  auto target = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "target", targetType, ArrayRef<StringRef>{});
  Block* targetEntry = target.addEntryBlock();
  builder.setInsertionPointToEnd(targetEntry);
  auto retVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 42));
  builder.create<ReturnOp>(builder.getUnknownLoc(), retVal.getResult());

  // Define caller function
  auto callerType = builder.getFunctionType({}, {f64Type});
  auto caller = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", callerType,
      ArrayRef<StringRef>{});
  Block* callerEntry = caller.addEntryBlock();
  builder.setInsertionPointToEnd(callerEntry);

  // Call target but expect f64 result (mismatch with i64 return type)
  auto call = builder.create<CallOp>(
      builder.getUnknownLoc(), "target",
      TypeRange{f64Type}, ValueRange{});
  builder.create<ReturnOp>(builder.getUnknownLoc(), call.getResult());

  EXPECT_FALSE(runPass(module));
  EXPECT_TRUE(hasDiagContaining("return type mismatch"));

  module->erase();
}

// ============== Unification Failure (Concrete Type Conflict) ==============

TEST_F(TypeInferencePassTest, UnificationFailureConcreteTypes) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);

  // Function that returns i64 but has a SubOp with f64 operands
  auto funcType = builder.getFunctionType({}, {i64Type});
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "__polang_entry", funcType,
      ArrayRef<StringRef>{});
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // SubOp with f64 lhs and i64 rhs - can't unify
  auto lhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 5.0));
  auto rhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 3));
  auto sub = builder.create<SubOp>(
      builder.getUnknownLoc(), f64Type, lhs.getResult(), rhs.getResult());
  builder.create<ReturnOp>(builder.getUnknownLoc(), sub.getResult());

  EXPECT_FALSE(runPass(module));
  EXPECT_TRUE(hasDiagContaining("operand type mismatch"));

  module->erase();
}

} // namespace
