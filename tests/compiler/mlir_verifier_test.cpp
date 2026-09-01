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

// ============== ReturnOp Verifier Tests ==============

TEST_F(VerifierTest, ReturnOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
  auto f64Type = builder.getF64Type();
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Return f64 from a function that expects i64
  auto val = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type, 1.0);
  builder.create<ReturnOp>(builder.getUnknownLoc(), val.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("returns"), std::string::npos);
}

TEST_F(VerifierTest, ReturnOpValueWhenVoid) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
  // Function with no return type (void)
  auto funcType = builder.getFunctionType({}, {});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  // Return a value from void function
  auto val = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type, 42);
  builder.create<ReturnOp>(builder.getUnknownLoc(), val.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("returns a value but function has no return type"),
            std::string::npos);
}

// ============== CallOp Verifier Tests ==============

TEST_F(VerifierTest, CallOpUndefinedFunction) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
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
  auto i64Type = builder.getI64Type();

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
  auto i64Type = builder.getI64Type();
  auto f64Type = builder.getF64Type();

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
      builder.getUnknownLoc(), f64Type, 1.0);
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
  auto i64Type = builder.getI64Type();
  auto f64Type = builder.getF64Type();
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type, 1);
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type, 2.0);

  // AddOp with mismatched types
  builder.create<AddOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("operand types must be compatible"), std::string::npos);
}

TEST_F(VerifierTest, SubOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
  auto f64Type = builder.getF64Type();
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type, 1);
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type, 2.0);

  builder.create<SubOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("operand types must be compatible"), std::string::npos);
}

TEST_F(VerifierTest, MulOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
  auto f64Type = builder.getF64Type();
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type, 1);
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type, 2.0);

  builder.create<MulOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("operand types must be compatible"), std::string::npos);
}

TEST_F(VerifierTest, DivOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
  auto f64Type = builder.getF64Type();
  auto funcType = builder.getFunctionType({}, {i64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type, 1);
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type, 2.0);

  builder.create<DivOp>(
      builder.getUnknownLoc(), i64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("operand types must be compatible"), std::string::npos);
}

TEST_F(VerifierTest, AddOpResultTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
  auto f64Type = builder.getF64Type();
  auto funcType = builder.getFunctionType({}, {f64Type});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type, 1);
  auto rhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type, 2);

  // AddOp with i64 operands but f64 result type
  builder.create<AddOp>(
      builder.getUnknownLoc(), f64Type, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("result type must be compatible"), std::string::npos);
}

// ============== CmpOp Verifier Tests ==============

TEST_F(VerifierTest, CmpOpTypeMismatch) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
  auto f64Type = builder.getF64Type();
  auto boolType = builder.getI1Type();
  auto funcType = builder.getFunctionType({}, {boolType});

  auto [module, func] = createModule("test", funcType);
  Block* entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto lhs = builder.create<ConstantIntegerOp>(
      builder.getUnknownLoc(), i64Type, 1);
  auto rhs = builder.create<ConstantFloatOp>(
      builder.getUnknownLoc(), f64Type, 2.0);

  builder.create<CmpOp>(
      builder.getUnknownLoc(), boolType,
      CmpPredicate::eq, lhs.getResult(), rhs.getResult());

  EXPECT_TRUE(failed(verify(*module)));
  EXPECT_NE(lastDiag.find("comparison operand types must be compatible"),
            std::string::npos);
}

TEST_F(VerifierTest, TupleTypeVerification) {
  OpBuilder builder(&context);
  auto i64Type = builder.getI64Type();
  auto validTuple = polang::TupleType::getChecked(
      [&] { return emitError(UnknownLoc::get(&context)); }, &context,
      ArrayRef<Type>{i64Type});
  EXPECT_TRUE(validTuple != nullptr);

  // Nested tuple is accepted
  auto nestedTuple = polang::TupleType::getChecked(
      [&] { return emitError(UnknownLoc::get(&context)); }, &context,
      ArrayRef<Type>{validTuple, i64Type});
  EXPECT_TRUE(nestedTuple != nullptr);

  // Non-Polang/non-MLIR standard type (e.g. NoneType) is rejected
  auto invalidTuple = polang::TupleType::getChecked(
      [&] { return emitError(UnknownLoc::get(&context)); }, &context,
      ArrayRef<Type>{builder.getNoneType()});
  EXPECT_TRUE(invalidTuple == nullptr);
  EXPECT_NE(lastDiag.find("not a valid Polang type or type parameter"),
            std::string::npos);
}

TEST_F(VerifierTest, TupleTypeSizeAndOffsetCalculation) {
  OpBuilder builder(&context);
  auto i1Type = builder.getI1Type();
  auto i8Type = builder.getI8Type();
  auto i16Type = builder.getI16Type();
  auto i32Type = builder.getI32Type();
  auto i64Type = builder.getI64Type();
  auto f32Type = builder.getF32Type();
  auto f64Type = builder.getF64Type();
  auto indexType = builder.getIndexType();

  // 1. Primitive type sizes
  EXPECT_EQ(polang::getTypeSize(i1Type), 1u);
  EXPECT_EQ(polang::getTypeSize(i8Type), 1u);
  EXPECT_EQ(polang::getTypeSize(i16Type), 2u);
  EXPECT_EQ(polang::getTypeSize(i32Type), 4u);
  EXPECT_EQ(polang::getTypeSize(i64Type), 8u);
  EXPECT_EQ(polang::getTypeSize(f32Type), 4u);
  EXPECT_EQ(polang::getTypeSize(f64Type), 8u);
  EXPECT_EQ(polang::getTypeSize(indexType), 8u);

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

  // 4. Flat tuple with sub-64-bit types (i8, i1, f32)
  auto sub64Tuple =
      polang::TupleType::get(&context, {i8Type, i1Type, f32Type});
  EXPECT_EQ(sub64Tuple.typeSize(), 24u);
  EXPECT_EQ(sub64Tuple.getNumSlots(), 3u);
  EXPECT_EQ(sub64Tuple.getElementOffsets(), (SmallVector<uint64_t>{0, 8, 16}));
  EXPECT_EQ(sub64Tuple.getElementSlotOffsets(),
            (SmallVector<uint64_t>{0, 1, 2}));

  // 5. Nested tuple (i64, (f64, i1), i32)
  auto innerTuple = polang::TupleType::get(&context, {f64Type, i1Type});
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

  // 6. Deeply nested tuple ((i64, f64), (i1, (i32, f32)))
  auto pair1 = polang::TupleType::get(&context, {i64Type, f64Type});
  auto pair2 = polang::TupleType::get(&context, {i32Type, f32Type});
  auto group = polang::TupleType::get(&context, {i1Type, pair2});
  EXPECT_EQ(pair1.typeSize(), 16u);
  EXPECT_EQ(pair2.typeSize(), 16u);
  EXPECT_EQ(group.typeSize(), 24u);
  auto deepTuple = polang::TupleType::get(&context, {pair1, group});
  EXPECT_EQ(deepTuple.typeSize(), 40u);
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

  // 8. Nested unspecialized generic tuple (i64, (type_param<"a">, f64), i1)
  auto genericInner = polang::TupleType::get(&context, {typeParamA, f64Type});
  auto genericNested =
      polang::TupleType::get(&context, {i64Type, genericInner, i1Type});
  EXPECT_EQ(genericNested.typeSize(), std::nullopt);
  EXPECT_EQ(genericNested.getNumSlots(), std::nullopt);
  EXPECT_EQ(genericNested.getElementOffset(0), 0u);
  EXPECT_EQ(genericNested.getElementOffset(1), 8u);
  EXPECT_EQ(genericNested.getElementOffset(2), std::nullopt);
  EXPECT_EQ(genericNested.getElementOffsets(), std::nullopt);
}

TEST_F(VerifierTest, TargetDataLayoutInteraction) {
  OpBuilder builder(&context);
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getI64Type();

  // 1. Default DataLayout (64-bit index)
  mlir::DataLayout defaultDL;
  EXPECT_EQ(defaultDL.getTypeSize(indexType), 8u);
  EXPECT_EQ(defaultDL.getTypeSizeInBits(indexType), 64u);
  EXPECT_EQ(polang::getTypeSize(indexType, &defaultDL), 8u);

  // 2. Custom 32-bit index DataLayout via module with DLTI spec
  auto loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module32 = ModuleOp::create(loc);
  auto entry = DataLayoutEntryAttr::get(
      mlir::IndexType::get(&context), builder.getI64IntegerAttr(32));
  auto spec = DataLayoutSpecAttr::get(
      &context, ArrayRef<DataLayoutEntryInterface>{entry});
  (*module32)->setAttr(DLTIDialect::kDataLayoutAttrName, spec);

  mlir::DataLayout dl32(*module32);
  EXPECT_EQ(dl32.getTypeSize(indexType), 4u);
  EXPECT_EQ(dl32.getTypeSizeInBits(indexType), 32u);
  EXPECT_EQ(polang::getTypeSize(indexType, &dl32), 4u);

  // 3. TupleType layout with 32-bit index and 64-bit alignment rule
  auto tupleWithIndices =
      polang::TupleType::get(&context, {indexType, indexType, i64Type});
  EXPECT_EQ(tupleWithIndices.typeSize(&dl32), 24u);
  EXPECT_EQ(tupleWithIndices.getNumSlots(&dl32), 3u);
  EXPECT_EQ(tupleWithIndices.getElementOffsets(&dl32),
            (SmallVector<uint64_t>{0, 8, 16}));
  EXPECT_EQ(tupleWithIndices.getElementSlotOffsets(&dl32),
            (SmallVector<uint64_t>{0, 1, 2}));
}

} // namespace
