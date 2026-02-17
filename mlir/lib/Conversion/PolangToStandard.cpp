//===- PolangToStandard.cpp - Polang to Standard lowering -------*- C++ -*-===//
//
// This file implements the lowering pass from Polang dialect to standard
// dialects (arith, func, scf, memref).
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Conversion/Passes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#pragma GCC diagnostic pop

using namespace mlir;
using namespace polang;

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PolangTypeConverter : public TypeConverter {
public:
  PolangTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](polang::IntegerType type) {
      // All integer types map to LLVM integer types (signedness is in ops)
      return mlir::IntegerType::get(type.getContext(), type.getWidth());
    });
    addConversion([](polang::FloatType type) {
      if (type.getWidth() == 32) {
        return (Type)Float32Type::get(type.getContext());
      }
      return (Type)Float64Type::get(type.getContext());
    });
    addConversion([](BoolType type) {
      return mlir::IntegerType::get(type.getContext(), 1);
    });
    addConversion([](polang::IndexType type) {
      return mlir::IndexType::get(type.getContext());
    });
    // Handle type parameters that weren't resolved by monomorphization.
    // Default to i64 as fallback (should not be reached in normal flow).
    addConversion([](TypeParamType type) -> Type {
      llvm::errs() << "warning: unresolved TypeParamType '" << type.getName()
                   << "' defaulting to i64\n";
      return mlir::IntegerType::get(type.getContext(), 64);
    });
  }
};

//===----------------------------------------------------------------------===//
// Constant Lowering
//===----------------------------------------------------------------------===//

struct ConstantIntegerOpLowering
    : public OpConversionPattern<ConstantIntegerOp> {
  using OpConversionPattern<ConstantIntegerOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantIntegerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor; // Unused, but required by MLIR interface
    Type resultType = op.getResult().getType();

    if (isa<polang::IndexType>(resultType)) {
      auto value = rewriter.create<arith::ConstantIndexOp>(
          op.getLoc(), op.getValue().getSExtValue());
      rewriter.replaceOp(op, value);
      return success();
    }

    unsigned width = 64; // Default width
    if (auto polangType = dyn_cast<polang::IntegerType>(resultType)) {
      width = polangType.getWidth();
    }

    auto intType = rewriter.getIntegerType(width);
    auto value = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), op.getValue().getSExtValue(), intType);
    rewriter.replaceOp(op, value);
    return success();
  }
};

struct ConstantFloatOpLowering : public OpConversionPattern<ConstantFloatOp> {
  using OpConversionPattern<ConstantFloatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantFloatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor; // Unused, but required by MLIR interface
    Type resultType = op.getResult().getType();
    mlir::FloatType floatType = rewriter.getF64Type(); // Default to f64

    if (auto polangType = dyn_cast<polang::FloatType>(resultType)) {
      if (polangType.getWidth() == 32) {
        floatType = rewriter.getF32Type();
      } else {
        floatType = rewriter.getF64Type();
      }
    }

    // getValue() returns APFloat from the attribute
    auto value = rewriter.create<arith::ConstantFloatOp>(
        op.getLoc(), op.getValue(), floatType);
    rewriter.replaceOp(op, value);
    return success();
  }
};

struct ConstantBoolOpLowering : public OpConversionPattern<ConstantBoolOp> {
  using OpConversionPattern<ConstantBoolOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor; // Unused, but required by MLIR interface
    auto i1Type = rewriter.getI1Type();
    auto value = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), op.getValue() ? 1 : 0, i1Type);
    rewriter.replaceOp(op, value);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arithmetic Lowering
//===----------------------------------------------------------------------===//

/// Check if the original Polang type is unsigned.
[[nodiscard]] bool isOrigTypeUnsigned(Type origType) noexcept {
  if (auto intType = dyn_cast<polang::IntegerType>(origType)) {
    return intType.isUnsigned();
  }
  if (auto indexType = dyn_cast<polang::IndexType>(origType)) {
    return indexType.isUnsigned();
  }
  return false;
}

struct AddOpLowering : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // After type conversion, integer/index types use AddIOp, floats use AddFOp
    if (isa<mlir::IntegerType, mlir::IndexType>(lhs.getType())) {
      const bool isUnsigned = isOrigTypeUnsigned(op.getLhs().getType());
      auto addOp = rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
      addOp->setAttr("polang.is_unsigned", rewriter.getBoolAttr(isUnsigned));
    } else {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, lhs, rhs);
    }
    return success();
  }
};

struct SubOpLowering : public OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // After type conversion, integer/index types use SubIOp, floats use SubFOp
    if (isa<mlir::IntegerType, mlir::IndexType>(lhs.getType())) {
      const bool isUnsigned = isOrigTypeUnsigned(op.getLhs().getType());
      auto subOp = rewriter.replaceOpWithNewOp<arith::SubIOp>(op, lhs, rhs);
      subOp->setAttr("polang.is_unsigned", rewriter.getBoolAttr(isUnsigned));
    } else {
      rewriter.replaceOpWithNewOp<arith::SubFOp>(op, lhs, rhs);
    }
    return success();
  }
};

struct MulOpLowering : public OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // After type conversion, integer/index types use MulIOp, floats use MulFOp
    if (isa<mlir::IntegerType, mlir::IndexType>(lhs.getType())) {
      const bool isUnsigned = isOrigTypeUnsigned(op.getLhs().getType());
      auto mulOp = rewriter.replaceOpWithNewOp<arith::MulIOp>(op, lhs, rhs);
      mulOp->setAttr("polang.is_unsigned", rewriter.getBoolAttr(isUnsigned));
    } else {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, lhs, rhs);
    }
    return success();
  }
};

/// Get or create the __polang_runtime_error function declaration in the module.
func::FuncOp getOrCreateRuntimeErrorFunc(ModuleOp module, OpBuilder& builder) {
  auto func = module.lookupSymbol<func::FuncOp>("__polang_runtime_error");
  if (func) {
    return func;
  }

  // Create function declaration: (ptr, i32, i32) -> ()
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto funcType = builder.getFunctionType(
      {ptrType, builder.getI32Type(), builder.getI32Type()}, {});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  func = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                      "__polang_runtime_error", funcType);
  func.setVisibility(SymbolTable::Visibility::Private);
  return func;
}

/// Get or create a global string constant, returning its symbol name.
/// The global is created in the module body as an LLVM::GlobalOp.
StringRef getOrCreateGlobalString(Location loc, OpBuilder& builder,
                                  ModuleOp module, StringRef name,
                                  StringRef value) {
  // Check if already exists
  if (module.lookupSymbol<LLVM::GlobalOp>(name)) {
    return name;
  }

  // Create global string constant
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto type = LLVM::LLVMArrayType::get(
      mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);

  std::string nullTerminated = (Twine(value) + Twine('\0')).str();
  builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                 LLVM::Linkage::Internal, name,
                                 builder.getStringAttr(nullTerminated));
  return name;
}

/// Extract line and column from a Location. Returns (0, 0) for unknown.
std::pair<int, int> extractLineColumn(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    return {static_cast<int>(fileLoc.getLine()),
            static_cast<int>(fileLoc.getColumn())};
  }
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (!fusedLoc.getLocations().empty()) {
      return extractLineColumn(fusedLoc.getLocations().front());
    }
  }
  return {0, 0};
}

/// Emit an scf.if guard that checks whether rhs is zero.
/// The then-block calls __polang_runtime_error and yields a dummy value.
/// The else-block is left empty for the caller to fill in the actual
/// arithmetic operation.
/// Returns {ifOp, isUnsigned} so the caller only needs to emit the
/// else-block operation (e.g. DivSIOp/RemSIOp).
[[nodiscard]] std::pair<scf::IfOp, bool>
emitIntegerZeroCheckGuard(Location loc, ConversionPatternRewriter& rewriter,
                          ModuleOp moduleOp, Value rhs, Type resultType,
                          Type origType) {
  getOrCreateRuntimeErrorFunc(moduleOp, rewriter);
  getOrCreateGlobalString(loc, rewriter, moduleOp,
                          "__polang_msg_integer_division_by_zero",
                          "integer division by zero");
  auto [line, col] = extractLineColumn(loc);
  const bool isUnsigned = isOrigTypeUnsigned(origType);

  // Create zero constant for comparison
  Value zeroConst;
  if (isa<mlir::IndexType>(rhs.getType())) {
    zeroConst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  } else {
    zeroConst = rewriter.create<arith::ConstantIntOp>(loc, 0, rhs.getType());
  }

  auto isZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               rhs, zeroConst);

  auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{resultType}, isZero,
                                         /*withElseRegion=*/true);

  // Then block (divisor is zero - error path)
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

    auto msgPtr = rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
        "__polang_msg_integer_division_by_zero");
    auto lineConst =
        rewriter.create<arith::ConstantIntOp>(loc, line, rewriter.getI32Type());
    auto colConst =
        rewriter.create<arith::ConstantIntOp>(loc, col, rewriter.getI32Type());
    rewriter.create<func::CallOp>(loc, "__polang_runtime_error", TypeRange{},
                                  ValueRange{msgPtr, lineConst, colConst});

    // Yield dummy value (unreachable - error handler calls exit)
    Value dummy;
    if (isa<mlir::IndexType>(resultType)) {
      dummy = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    } else {
      dummy = rewriter.create<arith::ConstantIntOp>(loc, 0, resultType);
    }
    rewriter.create<scf::YieldOp>(loc, ValueRange{dummy});
  }

  return {ifOp, isUnsigned};
}

struct DivOpLowering : public OpConversionPattern<DivOp> {
  using OpConversionPattern<DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto loc = op.getLoc();

    // Check the original type to determine signedness and type category
    auto origType = op.getLhs().getType();

    // Float division: no zero check (IEEE 754 produces inf/NaN)
    if (isa<polang::FloatType>(origType)) {
      rewriter.replaceOpWithNewOp<arith::DivFOp>(op, lhs, rhs);
      return success();
    }

    // For already-converted float types (fallback path)
    if (!isa<polang::IntegerType, polang::IndexType>(origType) &&
        !isa<mlir::IntegerType, mlir::IndexType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<arith::DivFOp>(op, lhs, rhs);
      return success();
    }

    // Integer division: insert zero check
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type resultType = lhs.getType();
    auto [ifOp, isUnsigned] = emitIntegerZeroCheckGuard(
        loc, rewriter, moduleOp, rhs, resultType, origType);

    // Else block (divisor is non-zero - normal division)
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());

      Value divResult;
      if (isUnsigned) {
        divResult = rewriter.create<arith::DivUIOp>(loc, lhs, rhs);
      } else {
        divResult = rewriter.create<arith::DivSIOp>(loc, lhs, rhs);
      }
      rewriter.create<scf::YieldOp>(loc, ValueRange{divResult});
    }

    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

struct NegOpLowering : public OpConversionPattern<NegOp> {
  using OpConversionPattern<NegOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto operand = adaptor.getOperand();
    auto origType = op.getOperand().getType();

    if (isa<polang::FloatType>(origType)) {
      rewriter.replaceOpWithNewOp<arith::NegFOp>(op, operand);
    } else {
      // Integer negation: 0 - x
      auto zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0,
                                                        operand.getType());
      auto subOp =
          rewriter.replaceOpWithNewOp<arith::SubIOp>(op, zero, operand);
      // Negation only applies to signed types
      subOp->setAttr("polang.is_unsigned", rewriter.getBoolAttr(false));
    }
    return success();
  }
};

struct RemOpLowering : public OpConversionPattern<RemOp> {
  using OpConversionPattern<RemOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto loc = op.getLoc();

    // Check the original type to determine signedness
    auto origType = op.getLhs().getType();

    // Remainder is only valid for integer types.
    // (Unlike DivOpLowering, no float path is needed because the type
    // checker rejects float operands for the modulo operator.)
    if (!isa<polang::IntegerType, polang::IndexType>(origType) &&
        !isa<mlir::IntegerType, mlir::IndexType>(lhs.getType())) {
      return failure();
    }

    // Insert zero check (shared guard with DivOpLowering)
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type resultType = lhs.getType();
    auto [ifOp, isUnsigned] = emitIntegerZeroCheckGuard(
        loc, rewriter, moduleOp, rhs, resultType, origType);

    // Else block (divisor is non-zero - normal remainder)
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());

      Value remResult;
      if (isUnsigned) {
        remResult = rewriter.create<arith::RemUIOp>(loc, lhs, rhs);
      } else {
        remResult = rewriter.create<arith::RemSIOp>(loc, lhs, rhs);
      }
      rewriter.create<scf::YieldOp>(loc, ValueRange{remResult});
    }

    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

struct NotOpLowering : public OpConversionPattern<NotOp> {
  using OpConversionPattern<NotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto operand = adaptor.getOperand();
    // Logical not: XOR with 1 (i1)
    auto one = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1,
                                                     rewriter.getI1Type());
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, operand, one);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Cast Lowering
//===----------------------------------------------------------------------===//

/// Lower integer to integer cast.
void lowerIntToIntCast(CastOp op, Value input, Type inputType, Type resultType,
                       Type origInputType,
                       ConversionPatternRewriter& rewriter) {
  auto inputIntType = cast<mlir::IntegerType>(inputType);
  auto resultIntType = cast<mlir::IntegerType>(resultType);
  unsigned inputWidth = inputIntType.getWidth();
  unsigned resultWidth = resultIntType.getWidth();

  if (inputWidth < resultWidth) {
    // Widening - check signedness of original input type
    bool isSigned = true; // Default to signed
    if (auto polangInt = dyn_cast<polang::IntegerType>(origInputType)) {
      isSigned = !polangInt.isUnsigned();
    }
    if (isSigned) {
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, resultType, input);
    } else {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, resultType, input);
    }
  } else if (inputWidth > resultWidth) {
    // Narrowing - truncate
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, resultType, input);
  } else {
    // Same width - just replace
    rewriter.replaceOp(op, input);
  }
}

/// Lower float to float cast.
void lowerFloatToFloatCast(CastOp op, Value input, Type inputType,
                           Type resultType,
                           ConversionPatternRewriter& rewriter) {
  unsigned inputWidth = inputType.getIntOrFloatBitWidth();
  unsigned resultWidth = resultType.getIntOrFloatBitWidth();

  if (inputWidth < resultWidth) {
    rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, resultType, input);
  } else if (inputWidth > resultWidth) {
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, resultType, input);
  } else {
    // Same width - just replace
    rewriter.replaceOp(op, input);
  }
}

/// Lower integer to float cast.
void lowerIntToFloatCast(CastOp op, Value input, Type resultType,
                         Type origInputType,
                         ConversionPatternRewriter& rewriter) {
  bool isSigned = true; // Default to signed
  if (auto polangInt = dyn_cast<polang::IntegerType>(origInputType)) {
    isSigned = !polangInt.isUnsigned();
  }
  if (isSigned) {
    rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, resultType, input);
  } else {
    rewriter.replaceOpWithNewOp<arith::UIToFPOp>(op, resultType, input);
  }
}

/// Lower float to integer cast using saturating intrinsics.
void lowerFloatToIntCast(CastOp op, Value input, Type inputType,
                         Type resultType, Type origResultType, Location loc,
                         ConversionPatternRewriter& rewriter) {
  // llvm.fptosi.sat / llvm.fptoui.sat clamp values to representable range
  bool isSigned = true; // Default to signed
  if (auto polangInt = dyn_cast<polang::IntegerType>(origResultType)) {
    isSigned = !polangInt.isUnsigned();
  }

  // Build intrinsic name: llvm.fptosi.sat.i<N>.f<M> or
  // llvm.fptoui.sat.i<N>.f<M>
  auto intType = cast<mlir::IntegerType>(resultType);
  unsigned intWidth = intType.getWidth();
  unsigned floatWidth = inputType.getIntOrFloatBitWidth();

  std::string intrinsicName =
      isSigned ? "llvm.fptosi.sat.i" : "llvm.fptoui.sat.i";
  intrinsicName += std::to_string(intWidth) + ".f" + std::to_string(floatWidth);

  auto intrinsicAttr = rewriter.getStringAttr(intrinsicName);
  auto callOp = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, resultType, intrinsicAttr, ValueRange{input});
  rewriter.replaceOp(op, callOp.getResults());
}

/// Lower index to integer cast (arith.index_cast or arith.index_castui).
void lowerIndexToIntCast(CastOp op, Value input, Type resultType,
                         Type origInputType,
                         ConversionPatternRewriter& rewriter) {
  bool isSigned = true;
  if (auto indexType = dyn_cast<polang::IndexType>(origInputType)) {
    isSigned = !indexType.isUnsigned();
  }
  if (isSigned) {
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, resultType, input);
  } else {
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(op, resultType, input);
  }
}

/// Lower integer to index cast (arith.index_cast or arith.index_castui).
void lowerIntToIndexCast(CastOp op, Value input, Type resultType,
                         Type origResultType,
                         ConversionPatternRewriter& rewriter) {
  bool isSigned = true;
  if (auto indexType = dyn_cast<polang::IndexType>(origResultType)) {
    isSigned = !indexType.isUnsigned();
  }
  if (isSigned) {
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, resultType, input);
  } else {
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(op, resultType, input);
  }
}

/// Lower index to float cast (index → i64 → float).
void lowerIndexToFloatCast(CastOp op, Value input, Type resultType,
                           Type origInputType, Location loc,
                           ConversionPatternRewriter& rewriter) {
  bool isSigned = true;
  if (auto indexType = dyn_cast<polang::IndexType>(origInputType)) {
    isSigned = !indexType.isUnsigned();
  }
  // First convert index to i64
  auto i64Type = rewriter.getI64Type();
  Value intVal;
  if (isSigned) {
    intVal = rewriter.create<arith::IndexCastOp>(loc, i64Type, input);
  } else {
    intVal = rewriter.create<arith::IndexCastUIOp>(loc, i64Type, input);
  }
  // Then convert i64 to float
  if (isSigned) {
    rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, resultType, intVal);
  } else {
    rewriter.replaceOpWithNewOp<arith::UIToFPOp>(op, resultType, intVal);
  }
}

/// Lower float to index cast (float → i64 → index).
void lowerFloatToIndexCast(CastOp op, Value input, Type inputType,
                           Type resultType, Type origResultType, Location loc,
                           ConversionPatternRewriter& rewriter) {
  bool isSigned = true;
  if (auto indexType = dyn_cast<polang::IndexType>(origResultType)) {
    isSigned = !indexType.isUnsigned();
  }
  // First convert float to i64 using saturating intrinsic
  auto i64Type = rewriter.getI64Type();
  unsigned floatWidth = inputType.getIntOrFloatBitWidth();
  std::string intrinsicName =
      isSigned ? "llvm.fptosi.sat.i64.f" : "llvm.fptoui.sat.i64.f";
  intrinsicName += std::to_string(floatWidth);

  auto intrinsicAttr = rewriter.getStringAttr(intrinsicName);
  auto callOp = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, i64Type, intrinsicAttr, ValueRange{input});
  // Then convert i64 to index
  if (isSigned) {
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, resultType,
                                                    callOp.getResult(0));
  } else {
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(op, resultType,
                                                      callOp.getResult(0));
  }
}

struct CastOpLowering : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Value input = adaptor.getInput();
    Type inputType = input.getType();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }

    // Get original types for signedness info
    Type origInputType = op.getInput().getType();
    Type origResultType = op.getResult().getType();

    // Determine type categories
    const bool inputIsInt = isa<mlir::IntegerType>(inputType);
    const bool resultIsInt = isa<mlir::IntegerType>(resultType);
    const bool inputIsIndex = isa<mlir::IndexType>(inputType);
    const bool resultIsIndex = isa<mlir::IndexType>(resultType);
    const bool inputIsFloat = !inputIsInt && !inputIsIndex;
    const bool resultIsFloat = !resultIsInt && !resultIsIndex;

    if (inputIsIndex && resultIsIndex) {
      // index → index: noop (both isize and usize are MLIR index)
      rewriter.replaceOp(op, input);
    } else if (inputIsIndex && resultIsInt) {
      lowerIndexToIntCast(op, input, resultType, origInputType, rewriter);
    } else if (inputIsInt && resultIsIndex) {
      lowerIntToIndexCast(op, input, resultType, origResultType, rewriter);
    } else if (inputIsIndex && resultIsFloat) {
      lowerIndexToFloatCast(op, input, resultType, origInputType, op.getLoc(),
                            rewriter);
    } else if (inputIsFloat && resultIsIndex) {
      lowerFloatToIndexCast(op, input, inputType, resultType, origResultType,
                            op.getLoc(), rewriter);
    } else if (inputIsInt && resultIsInt) {
      lowerIntToIntCast(op, input, inputType, resultType, origInputType,
                        rewriter);
    } else if (inputIsFloat && resultIsFloat) {
      lowerFloatToFloatCast(op, input, inputType, resultType, rewriter);
    } else if (inputIsInt && resultIsFloat) {
      lowerIntToFloatCast(op, input, resultType, origInputType, rewriter);
    } else {
      lowerFloatToIntCast(op, input, inputType, resultType, origResultType,
                          op.getLoc(), rewriter);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Comparison Lowering
//===----------------------------------------------------------------------===//

/// Convert a Polang comparison predicate to an MLIR floating-point predicate.
[[nodiscard]] arith::CmpFPredicate
convertToFloatPredicate(CmpPredicate pred) noexcept {
  switch (pred) {
  case CmpPredicate::eq:
    return arith::CmpFPredicate::OEQ;
  case CmpPredicate::ne:
    return arith::CmpFPredicate::ONE;
  case CmpPredicate::lt:
    return arith::CmpFPredicate::OLT;
  case CmpPredicate::le:
    return arith::CmpFPredicate::OLE;
  case CmpPredicate::gt:
    return arith::CmpFPredicate::OGT;
  case CmpPredicate::ge:
    return arith::CmpFPredicate::OGE;
  }
  llvm_unreachable("Unknown CmpPredicate");
}

/// Convert a Polang comparison predicate to an MLIR integer predicate.
/// \param pred The Polang comparison predicate.
/// \param isUnsigned Whether the integer type is unsigned.
[[nodiscard]] arith::CmpIPredicate
convertToIntPredicate(CmpPredicate pred, bool isUnsigned) noexcept {
  switch (pred) {
  case CmpPredicate::eq:
    return arith::CmpIPredicate::eq;
  case CmpPredicate::ne:
    return arith::CmpIPredicate::ne;
  case CmpPredicate::lt:
    return isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
  case CmpPredicate::le:
    return isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
  case CmpPredicate::gt:
    return isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
  case CmpPredicate::ge:
    return isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge;
  }
  llvm_unreachable("Unknown CmpPredicate");
}

struct CmpOpLowering : public OpConversionPattern<CmpOp> {
  using OpConversionPattern<CmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // Check the original type to determine signedness
    auto origType = op.getLhs().getType();

    if (isa<polang::FloatType>(origType)) {
      auto pred = convertToFloatPredicate(op.getPredicate());
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, pred, lhs, rhs);
    } else if (auto intType = dyn_cast<polang::IntegerType>(origType)) {
      auto pred =
          convertToIntPredicate(op.getPredicate(), intType.isUnsigned());
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, pred, lhs, rhs);
    } else if (auto indexType = dyn_cast<polang::IndexType>(origType)) {
      auto pred =
          convertToIntPredicate(op.getPredicate(), indexType.isUnsigned());
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, pred, lhs, rhs);
    } else if (isa<mlir::IntegerType, mlir::IndexType>(lhs.getType())) {
      // Fallback for already converted types - assume signed
      auto pred =
          convertToIntPredicate(op.getPredicate(), /*isUnsigned=*/false);
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, pred, lhs, rhs);
    } else {
      // Fallback to float comparison
      auto pred = convertToFloatPredicate(op.getPredicate());
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, pred, lhs, rhs);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Function Lowering
//===----------------------------------------------------------------------===//

struct GenericFuncOpLowering : public OpConversionPattern<GenericFuncOp> {
  using OpConversionPattern<GenericFuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GenericFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    // GenericFuncOp should have been monomorphized already.
    // Erase any remaining instances as a safety net.
    rewriter.eraseOp(op);
    return success();
  }
};

struct InstantiateOpLowering : public OpConversionPattern<InstantiateOp> {
  using OpConversionPattern<InstantiateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstantiateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    (void)rewriter;
    // InstantiateOp should have been replaced by CallOp during
    // monomorphization. If we reach here, something went wrong.
    return op.emitOpError(
        "was not resolved during monomorphization; this is a compiler bug");
  }
};

struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor; // Unused, but required by MLIR interface
    auto funcType = op.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(
        funcType.getNumInputs());

    const auto* typeConverter = getTypeConverter();
    for (size_t i = 0; i < funcType.getNumInputs(); ++i) {
      signatureConversion.addInputs(
          i, typeConverter->convertType(funcType.getInput(i)));
    }

    SmallVector<Type> resultTypes;
    if (failed(
            typeConverter->convertTypes(funcType.getResults(), resultTypes))) {
      return failure();
    }

    auto newFuncType = rewriter.getFunctionType(
        signatureConversion.getConvertedTypes(), resultTypes);

    auto newFunc = rewriter.create<func::FuncOp>(op.getLoc(), op.getSymName(),
                                                 newFuncType);

    // If the FuncOp has no body (extern declaration), set private visibility
    // and skip body conversion. This happens in incremental mode for
    // previously compiled functions.
    if (op.getBody().empty()) {
      newFunc.setVisibility(SymbolTable::Visibility::Private);
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), resultTypes,
                                              adaptor.getOperands());
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Control Flow Lowering
//===----------------------------------------------------------------------===//

struct IfOpLowering : public OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }

    // Create scf::IfOp with empty regions (no withElseRegion to avoid
    // auto-created blocks)
    auto scfIf = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange{resultType},
                                            adaptor.getCondition());

    // Erase the auto-generated empty blocks if any
    if (!scfIf.getThenRegion().empty()) {
      rewriter.eraseBlock(&scfIf.getThenRegion().front());
    }
    if (!scfIf.getElseRegion().empty()) {
      rewriter.eraseBlock(&scfIf.getElseRegion().front());
    }

    // Move then region
    rewriter.inlineRegionBefore(op.getThenRegion(), scfIf.getThenRegion(),
                                scfIf.getThenRegion().end());
    // Move else region
    rewriter.inlineRegionBefore(op.getElseRegion(), scfIf.getElseRegion(),
                                scfIf.getElseRegion().end());

    rewriter.replaceOp(op, scfIf.getResults());
    return success();
  }
};

struct YieldOpLowering : public OpConversionPattern<YieldOp> {
  using OpConversionPattern<YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Variable Operations Lowering
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Global Variable Operations Lowering
//===----------------------------------------------------------------------===//

struct GlobalOpLowering : public OpConversionPattern<GlobalOp> {
  using OpConversionPattern<GlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    auto elementType = getTypeConverter()->convertType(op.getType());
    if (!elementType) {
      return failure();
    }

    // Lower to memref.global with 0-d memref type.
    // finalize-memref-to-llvm (already in the pipeline) handles the rest.
    auto memrefType = MemRefType::get({}, elementType);

    // External globals: no initial_value, public visibility
    // Non-external: zero-initializer, public visibility
    Attribute initialValue;
    if (!op.getIsExternal()) {
      if (auto intTy = dyn_cast<mlir::IntegerType>(elementType)) {
        initialValue =
            DenseElementsAttr::get(RankedTensorType::get({}, intTy),
                                   rewriter.getIntegerAttr(intTy, 0));
      } else if (auto floatTy = dyn_cast<mlir::FloatType>(elementType)) {
        initialValue =
            DenseElementsAttr::get(RankedTensorType::get({}, floatTy),
                                   rewriter.getFloatAttr(floatTy, 0.0));
      } else if (isa<mlir::IndexType>(elementType)) {
        initialValue = DenseElementsAttr::get(
            RankedTensorType::get({}, elementType),
            IntegerAttr::get(mlir::IndexType::get(rewriter.getContext()), 0));
      }
    }

    auto memrefGlobal = rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op, op.getSymName(),
        /*sym_visibility=*/rewriter.getStringAttr("public"), memrefType,
        initialValue, /*constant=*/false, /*alignment=*/IntegerAttr());
    (void)memrefGlobal;
    return success();
  }
};

struct GlobalLoadOpLowering : public OpConversionPattern<GlobalLoadOp> {
  using OpConversionPattern<GlobalLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GlobalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    auto loc = op.getLoc();

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }

    // Lower to memref.get_global + memref.load (0-d memref, empty indices)
    auto memrefType = MemRefType::get({}, resultType);
    auto getGlobal = rewriter.create<memref::GetGlobalOp>(loc, memrefType,
                                                          op.getGlobalName());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, getGlobal, ValueRange{});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Print Operation Lowering
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<PrintOp> {
  using OpConversionPattern<PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor; // Unused, but required by MLIR interface
    // For now, just erase the print operation
    // In a full implementation, this would lower to a runtime call
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct PolangToStandardPass
    : public PassWrapper<PolangToStandardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolangToStandardPass)

  [[nodiscard]] StringRef getArgument() const final {
    return "convert-polang-to-standard";
  }
  [[nodiscard]] StringRef getDescription() const final {
    return "Lower Polang dialect to standard dialects";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  /// Record of a global's init function created by the pre-step.
  struct InitFuncRecord {
    std::string globalName;
    std::string initFuncName;
  };

  /// Pre-step: extract init regions from GlobalOps into polang::FuncOps.
  /// Returns records for the post-step. After this, all GlobalOps have
  /// empty init regions and can be lowered by the normal conversion pattern.
  SmallVector<InitFuncRecord> extractGlobalInitRegions(ModuleOp moduleOp) {
    SmallVector<InitFuncRecord> records;
    OpBuilder builder(&getContext());

    for (auto globalOp :
         llvm::make_early_inc_range(moduleOp.getOps<GlobalOp>())) {
      if (globalOp.getInitializer().empty()) {
        continue;
      }

      std::string globalName = globalOp.getSymName().str();
      std::string initFuncName = "__polang_init_" + globalName;
      Type globalType = globalOp.getType();

      // Create polang::FuncOp that returns the init value
      auto funcType = builder.getFunctionType({}, {globalType});
      builder.setInsertionPoint(globalOp);
      auto initFunc =
          builder.create<FuncOp>(globalOp.getLoc(), initFuncName, funcType);

      // Move init region into the FuncOp body
      initFunc.getBody().takeBody(globalOp.getInitializer());

      // Replace YieldGlobalOp terminator with polang::ReturnOp
      auto& funcBlock = initFunc.getBody().front();
      auto yieldOp = cast<YieldGlobalOp>(funcBlock.getTerminator());
      builder.setInsertionPoint(yieldOp);
      builder.create<ReturnOp>(yieldOp.getLoc(), yieldOp.getValue());
      yieldOp.erase();

      records.push_back({globalName, initFuncName});
    }

    return records;
  }

  /// Post-step: transform converted init functions to void (store inside)
  /// and insert calls in the entry function.
  void finalizeGlobalInitFunctions(ModuleOp moduleOp,
                                   const SmallVector<InitFuncRecord>& records) {
    if (records.empty()) {
      return;
    }

    auto* ctx = &getContext();
    OpBuilder builder(ctx);

    // Phase A: Transform each init function to store + void return
    for (const auto& record : records) {
      auto initFunc = moduleOp.lookupSymbol<func::FuncOp>(record.initFuncName);
      if (!initFunc) {
        continue;
      }

      // Find the return op in the function
      auto& funcBlock = initFunc.getBody().front();
      auto returnOp = cast<func::ReturnOp>(funcBlock.getTerminator());
      Value retVal = returnOp.getOperand(0);
      Type valType = retVal.getType();

      // Insert memref.get_global + memref.store before the return
      builder.setInsertionPoint(returnOp);
      auto memrefType = MemRefType::get({}, valType);
      auto getGlobal = builder.create<memref::GetGlobalOp>(
          returnOp.getLoc(), memrefType, record.globalName);
      builder.create<memref::StoreOp>(returnOp.getLoc(), retVal, getGlobal,
                                      ValueRange{});

      // Replace return with void return
      builder.setInsertionPoint(returnOp);
      builder.create<func::ReturnOp>(returnOp.getLoc());
      returnOp.erase();

      // Update function type to void
      initFunc.setFunctionType(FunctionType::get(ctx, {}, {}));

      // Set private visibility (these are internal helpers)
      initFunc.setVisibility(SymbolTable::Visibility::Private);
    }

    // Phase B: Insert calls in entry function (in module order)
    // Find the entry function by name convention:
    //   - REPL: __polang_eval_N
    //   - Compiler: __polang_entry
    func::FuncOp entryFunc;
    for (auto func : moduleOp.getOps<func::FuncOp>()) {
      auto name = func.getSymName();
      if (!func.getBody().empty() &&
          (name.starts_with("__polang_eval_") || name == "__polang_entry")) {
        entryFunc = func;
        break;
      }
    }
    if (!entryFunc) {
      return;
    }

    auto& entryBlock = entryFunc.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);

    for (const auto& record : records) {
      builder.create<func::CallOp>(entryFunc.getLoc(), record.initFuncName,
                                   TypeRange{});
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Pre-step: extract init regions into polang::FuncOps
    auto initRecords = extractGlobalInitRegions(moduleOp);

    // Main conversion
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           scf::SCFDialect, memref::MemRefDialect,
                           LLVM::LLVMDialect>();
    target.addIllegalDialect<PolangDialect>();

    PolangTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());

    patterns
        .add<ConstantIntegerOpLowering, ConstantFloatOpLowering,
             ConstantBoolOpLowering, AddOpLowering, SubOpLowering,
             MulOpLowering, DivOpLowering, RemOpLowering, NegOpLowering,
             NotOpLowering, CastOpLowering, CmpOpLowering,
             GenericFuncOpLowering, InstantiateOpLowering, FuncOpLowering,
             CallOpLowering, ReturnOpLowering, IfOpLowering, YieldOpLowering,
             GlobalOpLowering, GlobalLoadOpLowering, PrintOpLowering>(
            typeConverter, &getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Post-step: finalize init functions and insert calls
    finalizeGlobalInitFunctions(moduleOp, initRecords);
  }
};

} // namespace

std::unique_ptr<Pass> polang::createPolangToStandardPass() {
  return std::make_unique<PolangToStandardPass>();
}

void polang::registerPolangConversionPasses() {
  PassRegistration<PolangToStandardPass>();
}
