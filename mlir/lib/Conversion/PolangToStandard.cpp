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
    // Handle type variables that weren't resolved - apply defaults based on
    // kind
    addConversion([](TypeVarType type) -> Type {
      auto* ctx = type.getContext();
      switch (type.getKind()) {
      case TypeVarKind::Integer:
        // Default integer type variables to i64
        return mlir::IntegerType::get(ctx, 64);
      case TypeVarKind::Float:
        // Default float type variables to f64
        return Float64Type::get(ctx);
      case TypeVarKind::Any:
        // Generic type vars default to i64 (legacy behavior)
        return mlir::IntegerType::get(ctx, 64);
      }
      llvm_unreachable("Unknown TypeVarKind");
    });
    // Handle mutable reference types - convert to memref
    addConversion([this](MutRefType type) -> Type {
      Type elemType = convertType(type.getElementType());
      if (!elemType) {
        return nullptr;
      }
      return MemRefType::get({}, elemType);
    });
    // Handle immutable reference types - also convert to memref
    // (immutability is enforced at the type checker level)
    addConversion([this](RefType type) -> Type {
      Type elemType = convertType(type.getElementType());
      if (!elemType) {
        return nullptr;
      }
      return MemRefType::get({}, elemType);
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
    unsigned width = 64; // Default width

    if (auto polangType = dyn_cast<polang::IntegerType>(resultType)) {
      width = polangType.getWidth();
    } else if (auto typeVar = dyn_cast<TypeVarType>(resultType)) {
      // Type variable - use default width based on kind
      // Integer kind defaults to 64, which is already set
      (void)typeVar;
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
    } else if (auto typeVar = dyn_cast<TypeVarType>(resultType)) {
      // Type variable - float kind defaults to f64
      (void)typeVar;
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

struct AddOpLowering : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // After type conversion, we have mlir::IntegerType, not polang::IntegerType
    if (isa<mlir::IntegerType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
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

    // After type conversion, we have mlir::IntegerType, not polang::IntegerType
    if (isa<mlir::IntegerType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, lhs, rhs);
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

    // After type conversion, we have mlir::IntegerType, not polang::IntegerType
    if (isa<mlir::IntegerType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, lhs, rhs);
    } else {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, lhs, rhs);
    }
    return success();
  }
};

struct DivOpLowering : public OpConversionPattern<DivOp> {
  using OpConversionPattern<DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    // Check the original type to determine signedness
    auto origType = op.getLhs().getType();
    if (isa<polang::FloatType>(origType)) {
      rewriter.replaceOpWithNewOp<arith::DivFOp>(op, lhs, rhs);
    } else if (auto intType = dyn_cast<polang::IntegerType>(origType)) {
      if (intType.isUnsigned()) {
        rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, lhs, rhs);
      } else {
        rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, lhs, rhs);
      }
    } else if (isa<mlir::IntegerType>(lhs.getType())) {
      // Fallback for already converted types - assume signed
      rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, lhs, rhs);
    } else {
      rewriter.replaceOpWithNewOp<arith::DivFOp>(op, lhs, rhs);
    }
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

    // Determine if types are integer or float
    const bool inputIsInt = isa<mlir::IntegerType>(inputType);
    const bool resultIsInt = isa<mlir::IntegerType>(resultType);

    if (inputIsInt && resultIsInt) {
      lowerIntToIntCast(op, input, inputType, resultType, origInputType,
                        rewriter);
    } else if (!inputIsInt && !resultIsInt) {
      lowerFloatToFloatCast(op, input, inputType, resultType, rewriter);
    } else if (inputIsInt && !resultIsInt) {
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
    } else if (isa<mlir::IntegerType>(lhs.getType())) {
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

struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor; // Unused, but required by MLIR interface
    // Skip polymorphic functions - they are templates that should not be
    // lowered. Only their specialized (monomorphized) versions are lowered.
    if (op->hasAttr("polang.polymorphic")) {
      rewriter.eraseOp(op);
      return success();
    }

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

struct AllocaOpLowering : public OpConversionPattern<AllocaOp> {
  using OpConversionPattern<AllocaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor; // Unused, but required by MLIR interface
    auto elementType = getTypeConverter()->convertType(op.getElementType());
    if (!elementType) {
      return failure();
    }

    auto memRefType = MemRefType::get({}, elementType);
    rewriter.replaceOpWithNewOp<memref::AllocaOp>(op, memRefType);
    return success();
  }
};

struct LoadOpLowering : public OpConversionPattern<LoadOp> {
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getRef());
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<StoreOp> {
  using OpConversionPattern<StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, adaptor.getValue(),
                                                 adaptor.getRef());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reference Operations Lowering
//===----------------------------------------------------------------------===//

struct MutRefCreateOpLowering : public OpConversionPattern<MutRefCreateOp> {
  using OpConversionPattern<MutRefCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MutRefCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Convert the result type (mutref -> memref)
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }
    auto memRefType = cast<MemRefType>(resultType);

    // Allocate memory
    auto alloca = rewriter.create<memref::AllocaOp>(op.getLoc(), memRefType);

    // Store the initial value
    rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getInitialValue(),
                                     alloca);

    // Replace the op with the allocated memref
    rewriter.replaceOp(op, alloca.getResult());
    return success();
  }
};

struct MutRefDerefOpLowering : public OpConversionPattern<MutRefDerefOp> {
  using OpConversionPattern<MutRefDerefOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MutRefDerefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Dereference is just a memref.load
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getRef());
    return success();
  }
};

struct MutRefStoreOpLowering : public OpConversionPattern<MutRefStoreOp> {
  using OpConversionPattern<MutRefStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MutRefStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Store through mutable reference is just memref.store
    rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getValue(),
                                     adaptor.getRef());
    // The store operation returns the stored value
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

struct RefCreateOpLowering : public OpConversionPattern<RefCreateOp> {
  using OpConversionPattern<RefCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RefCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Creating an immutable reference from a mutable reference is a no-op
    // at the memref level (both are memref). The immutability is enforced
    // at the type checker level.
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

struct RefDerefOpLowering : public OpConversionPattern<RefDerefOp> {
  using OpConversionPattern<RefDerefOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RefDerefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Dereference immutable reference is just a memref.load
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getRef());
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

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           scf::SCFDialect, memref::MemRefDialect,
                           LLVM::LLVMDialect>();
    target.addIllegalDialect<PolangDialect>();

    PolangTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());

    patterns.add<ConstantIntegerOpLowering, ConstantFloatOpLowering,
                 ConstantBoolOpLowering, AddOpLowering, SubOpLowering,
                 MulOpLowering, DivOpLowering, CastOpLowering, CmpOpLowering,
                 FuncOpLowering, CallOpLowering, ReturnOpLowering, IfOpLowering,
                 YieldOpLowering, AllocaOpLowering, LoadOpLowering,
                 StoreOpLowering, MutRefCreateOpLowering, MutRefDerefOpLowering,
                 MutRefStoreOpLowering, RefCreateOpLowering, RefDerefOpLowering,
                 PrintOpLowering>(typeConverter, &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> polang::createPolangToStandardPass() {
  return std::make_unique<PolangToStandardPass>();
}

void polang::registerPolangConversionPasses() {
  PassRegistration<PolangToStandardPass>();
}
