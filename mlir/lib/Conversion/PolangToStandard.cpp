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
    auto polangType = mlir::cast<polang::IntegerType>(op.getResult().getType());
    auto intType = rewriter.getIntegerType(polangType.getWidth());
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
    auto polangType = mlir::cast<polang::FloatType>(op.getResult().getType());
    mlir::FloatType floatType;
    if (polangType.getWidth() == 32) {
      floatType = rewriter.getF32Type();
    } else {
      floatType = rewriter.getF64Type();
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
// Comparison Lowering
//===----------------------------------------------------------------------===//

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
      // Float comparison
      arith::CmpFPredicate pred = arith::CmpFPredicate::OEQ;
      switch (op.getPredicate()) {
      case CmpPredicate::eq:
        pred = arith::CmpFPredicate::OEQ;
        break;
      case CmpPredicate::ne:
        pred = arith::CmpFPredicate::ONE;
        break;
      case CmpPredicate::lt:
        pred = arith::CmpFPredicate::OLT;
        break;
      case CmpPredicate::le:
        pred = arith::CmpFPredicate::OLE;
        break;
      case CmpPredicate::gt:
        pred = arith::CmpFPredicate::OGT;
        break;
      case CmpPredicate::ge:
        pred = arith::CmpFPredicate::OGE;
        break;
      }
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, pred, lhs, rhs);
    } else if (auto intType = dyn_cast<polang::IntegerType>(origType)) {
      // Integer comparison - choose signed or unsigned predicate
      bool isUnsigned = intType.isUnsigned();
      arith::CmpIPredicate pred = arith::CmpIPredicate::eq;
      switch (op.getPredicate()) {
      case CmpPredicate::eq:
        pred = arith::CmpIPredicate::eq;
        break;
      case CmpPredicate::ne:
        pred = arith::CmpIPredicate::ne;
        break;
      case CmpPredicate::lt:
        pred = isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
        break;
      case CmpPredicate::le:
        pred = isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
        break;
      case CmpPredicate::gt:
        pred = isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
        break;
      case CmpPredicate::ge:
        pred = isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge;
        break;
      }
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, pred, lhs, rhs);
    } else if (isa<mlir::IntegerType>(lhs.getType())) {
      // Fallback for already converted types - assume signed
      arith::CmpIPredicate pred = arith::CmpIPredicate::eq;
      switch (op.getPredicate()) {
      case CmpPredicate::eq:
        pred = arith::CmpIPredicate::eq;
        break;
      case CmpPredicate::ne:
        pred = arith::CmpIPredicate::ne;
        break;
      case CmpPredicate::lt:
        pred = arith::CmpIPredicate::slt;
        break;
      case CmpPredicate::le:
        pred = arith::CmpIPredicate::sle;
        break;
      case CmpPredicate::gt:
        pred = arith::CmpIPredicate::sgt;
        break;
      case CmpPredicate::ge:
        pred = arith::CmpIPredicate::sge;
        break;
      }
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, pred, lhs, rhs);
    } else {
      // Fallback to float comparison
      arith::CmpFPredicate pred = arith::CmpFPredicate::OEQ;
      switch (op.getPredicate()) {
      case CmpPredicate::eq:
        pred = arith::CmpFPredicate::OEQ;
        break;
      case CmpPredicate::ne:
        pred = arith::CmpFPredicate::ONE;
        break;
      case CmpPredicate::lt:
        pred = arith::CmpFPredicate::OLT;
        break;
      case CmpPredicate::le:
        pred = arith::CmpFPredicate::OLE;
        break;
      case CmpPredicate::gt:
        pred = arith::CmpFPredicate::OGT;
        break;
      case CmpPredicate::ge:
        pred = arith::CmpFPredicate::OGE;
        break;
      }
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
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           scf::SCFDialect, memref::MemRefDialect>();
    target.addIllegalDialect<PolangDialect>();

    PolangTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());

    patterns.add<ConstantIntegerOpLowering, ConstantFloatOpLowering,
                 ConstantBoolOpLowering, AddOpLowering, SubOpLowering,
                 MulOpLowering, DivOpLowering, CmpOpLowering, FuncOpLowering,
                 CallOpLowering, ReturnOpLowering, IfOpLowering,
                 YieldOpLowering, AllocaOpLowering, LoadOpLowering,
                 StoreOpLowering, PrintOpLowering>(typeConverter,
                                                   &getContext());

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
