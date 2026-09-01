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

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

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
    context.getOrLoadDialect<DLTIDialect>();
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

TEST_F(VerifierTest, IntegerTypeVerification) {
  // Valid power-of-two widths >= 8
  EXPECT_TRUE(polang::IntegerType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 8u, Signedness::Signed) != nullptr);
  EXPECT_TRUE(polang::IntegerType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 16u, Signedness::Signed) != nullptr);
  EXPECT_TRUE(polang::IntegerType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 32u, Signedness::Signed) != nullptr);
  EXPECT_TRUE(polang::IntegerType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 64u, Signedness::Signed) != nullptr);

  // Non power-of-two widths rejected
  EXPECT_TRUE(polang::IntegerType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 24u, Signedness::Signed) == nullptr);
  EXPECT_NE(lastDiag.find("integer width must be a power of 2 >= 8"),
            std::string::npos);

  // Width < 8 rejected
  EXPECT_TRUE(polang::IntegerType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 4u, Signedness::Signed) == nullptr);
}

TEST_F(VerifierTest, FloatTypeVerification) {
  // Valid widths (32, 64)
  EXPECT_TRUE(polang::FloatType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 32u) != nullptr);
  EXPECT_TRUE(polang::FloatType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 64u) != nullptr);

  // Invalid widths rejected
  EXPECT_TRUE(polang::FloatType::getChecked(
                  [&] { return emitError(UnknownLoc::get(&context)); },
                  &context, 16u) == nullptr);
  EXPECT_NE(lastDiag.find("float width must be 32 or 64"), std::string::npos);
}

TEST_F(VerifierTest, TupleTypeVerification) {
  OpBuilder builder(&context);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto validTuple = polang::TupleType::getChecked(
      [&] { return emitError(UnknownLoc::get(&context)); }, &context,
      ArrayRef<Type>{i64Type});
  EXPECT_TRUE(validTuple != nullptr);

  // Nested tuple is accepted
  auto nestedTuple = polang::TupleType::getChecked(
      [&] { return emitError(UnknownLoc::get(&context)); }, &context,
      ArrayRef<Type>{validTuple, i64Type});
  EXPECT_TRUE(nestedTuple != nullptr);

  // Non-Polang type (e.g. NoneType) is rejected
  auto invalidTuple = polang::TupleType::getChecked(
      [&] { return emitError(UnknownLoc::get(&context)); }, &context,
      ArrayRef<Type>{builder.getNoneType()});
  EXPECT_TRUE(invalidTuple == nullptr);
  EXPECT_NE(lastDiag.find("not a valid Polang type or type parameter"),
            std::string::npos);
}

TEST_F(VerifierTest, TupleTypeSizeAndOffsetCalculation) {
  auto i8Type = polang::IntegerType::get(&context, 8, Signedness::Signed);
  auto i16Type = polang::IntegerType::get(&context, 16, Signedness::Signed);
  auto i32Type = polang::IntegerType::get(&context, 32, Signedness::Signed);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);
  auto f32Type = polang::FloatType::get(&context, 32);
  auto f64Type = polang::FloatType::get(&context, 64);
  auto boolType = BoolType::get(&context);
  auto isizeType = polang::IndexType::get(&context, Signedness::Signed);
  auto usizeType = polang::IndexType::get(&context, Signedness::Unsigned);

  // 1. Primitive type sizes
  EXPECT_EQ(i8Type.typeSize(), 1u);
  EXPECT_EQ(i16Type.typeSize(), 2u);
  EXPECT_EQ(i32Type.typeSize(), 4u);
  EXPECT_EQ(i64Type.typeSize(), 8u);
  EXPECT_EQ(f32Type.typeSize(), 4u);
  EXPECT_EQ(f64Type.typeSize(), 8u);
  EXPECT_EQ(boolType.typeSize(), 1u);
  EXPECT_EQ(isizeType.typeSize(), 8u);
  EXPECT_EQ(usizeType.typeSize(), 8u);

  EXPECT_EQ(polang::getTypeSize(i32Type), 4u);
  EXPECT_EQ(polang::getTypeSize(f64Type), 8u);

  // 2. Empty tuple ()
  auto emptyTuple = polang::TupleType::get(&context, {});
  EXPECT_EQ(emptyTuple.typeSize(), 0u);
  EXPECT_EQ(emptyTuple.getNumSlots(), 0u);
  ASSERT_TRUE(emptyTuple.getElementOffsets().has_value());
  EXPECT_TRUE(emptyTuple.getElementOffsets()->empty());
  ASSERT_TRUE(emptyTuple.getElementSlotOffsets().has_value());
  EXPECT_TRUE(emptyTuple.getElementSlotOffsets()->empty());

  // 3. Flat tuple (i64, f64)
  auto flatTuple = polang::TupleType::get(&context, {i64Type, f64Type});
  EXPECT_EQ(flatTuple.typeSize(), 16u);
  EXPECT_EQ(flatTuple.getNumSlots(), 2u);
  EXPECT_EQ(flatTuple.getElementOffset(0), 0u);
  EXPECT_EQ(flatTuple.getElementOffset(1), 8u);
  EXPECT_EQ(flatTuple.getElementSlotOffset(0), 0u);
  EXPECT_EQ(flatTuple.getElementSlotOffset(1), 1u);
  EXPECT_EQ(flatTuple.getElementOffsets(), (SmallVector<uint64_t>{0, 8}));
  EXPECT_EQ(flatTuple.getElementSlotOffsets(), (SmallVector<uint64_t>{0, 1}));

  // 4. Flat tuple with sub-64-bit types (i8, bool, f32)
  auto sub64Tuple =
      polang::TupleType::get(&context, {i8Type, boolType, f32Type});
  EXPECT_EQ(sub64Tuple.typeSize(), 24u);
  EXPECT_EQ(sub64Tuple.getNumSlots(), 3u);
  EXPECT_EQ(sub64Tuple.getElementOffsets(), (SmallVector<uint64_t>{0, 8, 16}));
  EXPECT_EQ(sub64Tuple.getElementSlotOffsets(),
            (SmallVector<uint64_t>{0, 1, 2}));

  // 5. Nested tuple (i64, (f64, bool), i32)
  auto innerTuple = polang::TupleType::get(&context, {f64Type, boolType});
  EXPECT_EQ(innerTuple.typeSize(), 16u);
  auto nestedTuple =
      polang::TupleType::get(&context, {i64Type, innerTuple, i32Type});
  EXPECT_EQ(nestedTuple.typeSize(), 32u);
  EXPECT_EQ(nestedTuple.getNumSlots(), 4u);
  EXPECT_EQ(nestedTuple.getElementOffset(0), 0u);
  EXPECT_EQ(nestedTuple.getElementOffset(1), 8u);
  EXPECT_EQ(nestedTuple.getElementOffset(2), 24u);
  EXPECT_EQ(nestedTuple.getElementSlotOffset(0), 0u);
  EXPECT_EQ(nestedTuple.getElementSlotOffset(1), 1u);
  EXPECT_EQ(nestedTuple.getElementSlotOffset(2), 3u);
  EXPECT_EQ(nestedTuple.getElementOffsets(), (SmallVector<uint64_t>{0, 8, 24}));
  EXPECT_EQ(nestedTuple.getElementSlotOffsets(),
            (SmallVector<uint64_t>{0, 1, 3}));

  // 6. Deeply nested tuple ((i64, f64), (bool, (i32, f32)))
  auto pair1 = polang::TupleType::get(&context, {i64Type, f64Type});
  auto pair2 = polang::TupleType::get(&context, {i32Type, f32Type});
  auto group = polang::TupleType::get(&context, {boolType, pair2});
  EXPECT_EQ(pair1.typeSize(), 16u);
  EXPECT_EQ(pair2.typeSize(), 16u);
  EXPECT_EQ(group.typeSize(), 24u); // bool at 0 (size 1), pair2 at 8 (size 16) -> total 24
  auto deepTuple = polang::TupleType::get(&context, {pair1, group});
  EXPECT_EQ(deepTuple.typeSize(), 40u); // pair1 at 0 (size 16), group at 16 (size 24) -> total 40
  EXPECT_EQ(deepTuple.getNumSlots(), 5u);
  EXPECT_EQ(deepTuple.getElementOffsets(), (SmallVector<uint64_t>{0, 16}));
  EXPECT_EQ(deepTuple.getElementSlotOffsets(), (SmallVector<uint64_t>{0, 2}));

  // 7. Unspecialized generic tuple (type_param<"a">, i64)
  auto typeParamA = polang::TypeParamType::get(&context, "a");
  auto genericTuple = polang::TupleType::get(&context, {typeParamA, i64Type});
  EXPECT_EQ(genericTuple.typeSize(), std::nullopt);
  EXPECT_EQ(genericTuple.getNumSlots(), std::nullopt);
  EXPECT_EQ(genericTuple.getElementOffset(0), 0u);
  EXPECT_EQ(genericTuple.getElementOffset(1), std::nullopt);
  EXPECT_EQ(genericTuple.getElementSlotOffset(0), 0u);
  EXPECT_EQ(genericTuple.getElementSlotOffset(1), std::nullopt);
  EXPECT_EQ(genericTuple.getElementOffsets(), std::nullopt);
  EXPECT_EQ(genericTuple.getElementSlotOffsets(), std::nullopt);

  // 8. Nested unspecialized generic tuple (i64, (type_param<"a">, f64), bool)
  auto genericInner = polang::TupleType::get(&context, {typeParamA, f64Type});
  auto genericNested =
      polang::TupleType::get(&context, {i64Type, genericInner, boolType});
  EXPECT_EQ(genericNested.typeSize(), std::nullopt);
  EXPECT_EQ(genericNested.getNumSlots(), std::nullopt);
  EXPECT_EQ(genericNested.getElementOffset(0), 0u);
  EXPECT_EQ(genericNested.getElementOffset(1), 8u);
  EXPECT_EQ(genericNested.getElementOffset(2), std::nullopt);
  EXPECT_EQ(genericNested.getElementOffsets(), std::nullopt);
}

TEST_F(VerifierTest, TargetDataLayoutInteraction) {
  OpBuilder builder(&context);
  auto isizeType = polang::IndexType::get(&context, Signedness::Signed);
  auto usizeType = polang::IndexType::get(&context, Signedness::Unsigned);
  auto i64Type = polang::IntegerType::get(&context, 64, Signedness::Signed);

  // 1. Default DataLayout (64-bit index)
  mlir::DataLayout defaultDL;
  EXPECT_EQ(defaultDL.getTypeSize(isizeType), 8u);
  EXPECT_EQ(defaultDL.getTypeSizeInBits(isizeType), 64u);
  EXPECT_EQ(isizeType.typeSize(&defaultDL), 8u);
  EXPECT_EQ(usizeType.typeSize(&defaultDL), 8u);

  // 2. Custom 32-bit index DataLayout via module with DLTI spec
  auto loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module32 = ModuleOp::create(loc);
  auto entry = DataLayoutEntryAttr::get(
      mlir::IndexType::get(&context), builder.getI64IntegerAttr(32));
  auto spec = DataLayoutSpecAttr::get(
      &context, ArrayRef<DataLayoutEntryInterface>{entry});
  (*module32)->setAttr(DLTIDialect::kDataLayoutAttrName, spec);

  mlir::DataLayout dl32(*module32);
  EXPECT_EQ(dl32.getTypeSize(isizeType), 4u);
  EXPECT_EQ(dl32.getTypeSizeInBits(isizeType), 32u);
  EXPECT_EQ(isizeType.typeSize(&dl32), 4u);
  EXPECT_EQ(usizeType.typeSize(&dl32), 4u);
  EXPECT_EQ(polang::getTypeSize(isizeType, &dl32), 4u);

  // 3. TupleType layout with 32-bit index and 64-bit alignment rule
  auto tupleWithIndices =
      polang::TupleType::get(&context, {isizeType, usizeType, i64Type});
  // Element 0: isize (4 bytes) at offset 0
  // Element 1: usize (4 bytes) at offset 8 (64-bit aligned)
  // Element 2: i64 (8 bytes) at offset 16 (64-bit aligned)
  // Total size: 24 bytes (3 slots)
  EXPECT_EQ(tupleWithIndices.typeSize(&dl32), 24u);
  EXPECT_EQ(tupleWithIndices.getNumSlots(&dl32), 3u);
  EXPECT_EQ(tupleWithIndices.getElementOffsets(&dl32),
            (SmallVector<uint64_t>{0, 8, 16}));
  EXPECT_EQ(tupleWithIndices.getElementSlotOffsets(&dl32),
            (SmallVector<uint64_t>{0, 1, 2}));
}

} // namespace
