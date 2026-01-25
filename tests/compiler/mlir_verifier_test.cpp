//===- mlir_verifier_test.cpp - Test MLIR verifiers --------*- C++ -*-===//
//
// Tests that exercise verifier error paths in PolangOps.cpp by constructing
// invalid MLIR programmatically (bypassing the type checker).
//
//===----------------------------------------------------------------------===//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

#pragma GCC diagnostic pop

#include <gtest/gtest.h>
#include <string>

using namespace mlir;
using namespace polang;

namespace {

class VerifierTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.getOrLoadDialect<PolangDialect>();
    // Capture diagnostics as strings
    diagHandler = context.getDiagEngine().registerHandler(
        [this](Diagnostic& diag) {
          lastDiag = diag.str();
          return success();
        });
  }

  void TearDown() override {
    context.getDiagEngine().eraseHandler(diagHandler);
  }

  /// Create a module with a function, returning the function's entry block.
  /// Caller can add operations to the block.
  std::pair<OwningOpRef<ModuleOp>, polang::FuncOp>
  createModule(StringRef funcName, FunctionType funcType) {
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());

    auto func = builder.create<polang::FuncOp>(
        builder.getUnknownLoc(), funcName, funcType, ArrayRef<StringRef>{});

    return {std::move(module), func};
  }

  MLIRContext context;
  std::string lastDiag;

private:
  DiagnosticEngine::HandlerID diagHandler;
};

// ============== IfOp Verifier Tests ==============

TEST_F(VerifierTest, IfOpEmptyThenRegion) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto boolType = BoolType::get(&context);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Create a bool constant for condition
  auto cond = builder.create<ConstantBoolOp>(
      builder.getUnknownLoc(), boolType, true);

  // Create IfOp - then region will be empty (no yield)
  auto ifOp = builder.create<polang::IfOp>(
      builder.getUnknownLoc(), i64Type, cond);

  // Leave then region empty (just the auto-created block)
  // But add yield to else region
  builder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
  auto elseVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 0));
  builder.create<YieldOp>(builder.getUnknownLoc(), elseVal.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("then region"), std::string::npos);
}

TEST_F(VerifierTest, IfOpEmptyElseRegion) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto boolType = BoolType::get(&context);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto cond = builder.create<ConstantBoolOp>(
      builder.getUnknownLoc(), boolType, true);

  auto ifOp = builder.create<polang::IfOp>(
      builder.getUnknownLoc(), i64Type, cond);

  // Add yield to then region
  builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
  auto thenVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  builder.create<YieldOp>(builder.getUnknownLoc(), thenVal.getResult());

  // Leave else region empty
  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("else region"), std::string::npos);
}

TEST_F(VerifierTest, IfOpYieldTypeMismatchThen) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto boolType = BoolType::get(&context);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto cond = builder.create<ConstantBoolOp>(
      builder.getUnknownLoc(), boolType, true);

  // IfOp expects i64 result
  auto ifOp = builder.create<polang::IfOp>(
      builder.getUnknownLoc(), i64Type, cond);

  // Then yields f64 (mismatch!)
  builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
  auto thenVal = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 1.0));
  builder.create<YieldOp>(builder.getUnknownLoc(), thenVal.getResult());

  // Else yields i64 (correct)
  builder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
  auto elseVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 0));
  builder.create<YieldOp>(builder.getUnknownLoc(), elseVal.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("then branch yields"), std::string::npos);
}

TEST_F(VerifierTest, IfOpYieldTypeMismatchElse) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto boolType = BoolType::get(&context);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto cond = builder.create<ConstantBoolOp>(
      builder.getUnknownLoc(), boolType, true);

  auto ifOp = builder.create<polang::IfOp>(
      builder.getUnknownLoc(), i64Type, cond);

  // Then yields i64 (correct)
  builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
  auto thenVal = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  builder.create<YieldOp>(builder.getUnknownLoc(), thenVal.getResult());

  // Else yields f64 (mismatch!)
  builder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
  auto elseVal = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 0.0));
  builder.create<YieldOp>(builder.getUnknownLoc(), elseVal.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("else branch yields"), std::string::npos);
}

// ============== ReturnOp Verifier Tests ==============

TEST_F(VerifierTest, ReturnOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Return f64 from a function that expects i64
  auto val = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 1.0));
  builder.create<ReturnOp>(builder.getUnknownLoc(), val.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("returns"), std::string::npos);
}

TEST_F(VerifierTest, ReturnOpValueWhenVoid) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  // Function with no return type (void)
  auto funcType = builder.getFunctionType({}, {});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Return a value from void function
  auto val = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 42));
  builder.create<ReturnOp>(builder.getUnknownLoc(), val.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("returns a value but function has no return type"),
            std::string::npos);
}

// ============== CallOp Verifier Tests ==============

TEST_F(VerifierTest, CallOpUndefinedFunction) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Call undefined function
  auto callOp = builder.create<CallOp>(
      builder.getUnknownLoc(), "nonexistent",
      TypeRange{i64Type}, ValueRange{});
  builder.create<ReturnOp>(builder.getUnknownLoc(), callOp.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("undefined function"), std::string::npos);
}

TEST_F(VerifierTest, CallOpArgCountMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);

  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Define target function with 1 parameter
  auto targetType = builder.getFunctionType({i64Type}, {i64Type});
  builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "target", targetType, ArrayRef<StringRef>{});

  // Define caller function
  auto callerType = builder.getFunctionType({}, {i64Type});
  auto caller = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "caller", callerType, ArrayRef<StringRef>{});
  Block* entry = caller.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Call with wrong number of arguments (0 instead of 1)
  auto callOp = builder.create<CallOp>(
      builder.getUnknownLoc(), "target",
      TypeRange{i64Type}, ValueRange{});
  builder.create<ReturnOp>(builder.getUnknownLoc(), callOp.getResult());

  EXPECT_TRUE(failed(verify(module)));
  EXPECT_NE(lastDiag.find("expects"), std::string::npos);
  EXPECT_NE(lastDiag.find("argument"), std::string::npos);
}

TEST_F(VerifierTest, CallOpArgTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);

  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Define target function expecting i64 parameter
  auto targetType = builder.getFunctionType({i64Type}, {i64Type});
  builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "target", targetType, ArrayRef<StringRef>{});

  // Define caller function
  auto callerType = builder.getFunctionType({}, {i64Type});
  auto caller = builder.create<polang::FuncOp>(
      builder.getUnknownLoc(), "caller", callerType, ArrayRef<StringRef>{});
  Block* entry = caller.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Pass f64 argument to function expecting i64
  auto val = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 1.0));
  auto callOp = builder.create<CallOp>(
      builder.getUnknownLoc(), "target",
      TypeRange{i64Type}, ValueRange{val.getResult()});
  builder.create<ReturnOp>(builder.getUnknownLoc(), callOp.getResult());

  EXPECT_TRUE(failed(verify(module)));
  EXPECT_NE(lastDiag.find("has type"), std::string::npos);
}

// ============== Arithmetic Op Verifier Tests ==============

TEST_F(VerifierTest, AddOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 2.0));

  // AddOp with mismatched types
  builder.create<AddOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("operand types must be compatible"), std::string::npos);
}

TEST_F(VerifierTest, SubOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 2.0));

  builder.create<SubOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("operand types must be compatible"), std::string::npos);
}

TEST_F(VerifierTest, MulOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 2.0));

  builder.create<MulOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("operand types must be compatible"), std::string::npos);
}

TEST_F(VerifierTest, DivOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 2.0));

  builder.create<DivOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("operand types must be compatible"), std::string::npos);
}

TEST_F(VerifierTest, AddOpResultTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto funcType = builder.getFunctionType({}, {f64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  auto rhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 2));

  // AddOp with i64 operands but f64 result type
  builder.create<AddOp>(
      builder.getUnknownLoc(), f64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("result type must be compatible"), std::string::npos);
}

// CastOp non-numeric type tests removed: ODS constraint (Polang_AnyNumericOrVar)
// prevents creating CastOp with BoolType, making the custom verifier's numeric
// checks unreachable.

// ============== CmpOp Verifier Tests ==============

TEST_F(VerifierTest, CmpOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto boolType = BoolType::get(&context);
  auto funcType = builder.getFunctionType({}, {boolType});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type,
      IntegerAttr::get(builder.getIntegerType(64), 1));
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type,
      FloatAttr::get(builder.getF64Type(), 2.0));

  builder.create<CmpOp>(
      builder.getUnknownLoc(), boolType,
      CmpPredicate::eq, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("comparison operand types must be compatible"),
            std::string::npos);
}

} // namespace
