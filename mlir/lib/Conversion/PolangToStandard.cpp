//===- PolangToStandard.cpp - Polang to Standard lowering -------*- C++ -*-===//
//
// This file implements the lowering pass from Polang dialect to standard
// dialects (arith, func, scf, memref).
//
//===----------------------------------------------------------------------===//

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
    addConversion(
        [](IntType type) { return IntegerType::get(type.getContext(), 64); });
    addConversion(
        [](DoubleType type) { return Float64Type::get(type.getContext()); });
    addConversion(
        [](BoolType type) { return IntegerType::get(type.getContext(), 1); });
  }
};

//===----------------------------------------------------------------------===//
// Constant Lowering
//===----------------------------------------------------------------------===//

struct ConstantIntOpLowering : public OpConversionPattern<ConstantIntOp> {
  using OpConversionPattern<ConstantIntOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto i64Type = rewriter.getI64Type();
    auto value = rewriter.create<arith::ConstantIntOp>(op.getLoc(),
                                                       op.getValue(), i64Type);
    rewriter.replaceOp(op, value);
    return success();
  }
};

struct ConstantDoubleOpLowering : public OpConversionPattern<ConstantDoubleOp> {
  using OpConversionPattern<ConstantDoubleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantDoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto f64Type = rewriter.getF64Type();
    auto value = rewriter.create<arith::ConstantFloatOp>(
        op.getLoc(), op.getValue(), f64Type);
    rewriter.replaceOp(op, value);
    return success();
  }
};

struct ConstantBoolOpLowering : public OpConversionPattern<ConstantBoolOp> {
  using OpConversionPattern<ConstantBoolOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
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

    if (isa<IntegerType>(lhs.getType())) {
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

    if (isa<IntegerType>(lhs.getType())) {
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

    if (isa<IntegerType>(lhs.getType())) {
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

    if (isa<IntegerType>(lhs.getType())) {
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

    if (isa<IntegerType>(lhs.getType())) {
      arith::CmpIPredicate pred;
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
      arith::CmpFPredicate pred;
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
    auto funcType = op.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(
        funcType.getNumInputs());

    const auto* typeConverter = getTypeConverter();
    for (size_t i = 0; i < funcType.getNumInputs(); ++i) {
      signatureConversion.addInputs(
          i, typeConverter->convertType(funcType.getInput(i)));
    }

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(funcType.getResults(), resultTypes)))
      return failure();

    auto newFuncType = rewriter.getFunctionType(
        signatureConversion.getConvertedTypes(), resultTypes);

    auto newFunc = rewriter.create<func::FuncOp>(op.getLoc(), op.getSymName(),
                                                 newFuncType);

    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *typeConverter,
                                           &signatureConversion)))
      return failure();

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
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

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
    if (!resultType)
      return failure();

    // Create scf::IfOp with empty regions (no withElseRegion to avoid
    // auto-created blocks)
    auto scfIf = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange{resultType},
                                            adaptor.getCondition());

    // Erase the auto-generated empty blocks if any
    if (!scfIf.getThenRegion().empty())
      rewriter.eraseBlock(&scfIf.getThenRegion().front());
    if (!scfIf.getElseRegion().empty())
      rewriter.eraseBlock(&scfIf.getElseRegion().front());

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
    auto elementType = getTypeConverter()->convertType(op.getElementType());
    if (!elementType)
      return failure();

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

  StringRef getArgument() const final { return "convert-polang-to-standard"; }
  StringRef getDescription() const final {
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

    patterns.add<ConstantIntOpLowering, ConstantDoubleOpLowering,
                 ConstantBoolOpLowering, AddOpLowering, SubOpLowering,
                 MulOpLowering, DivOpLowering, CmpOpLowering, FuncOpLowering,
                 CallOpLowering, ReturnOpLowering, IfOpLowering,
                 YieldOpLowering, AllocaOpLowering, LoadOpLowering,
                 StoreOpLowering, PrintOpLowering>(typeConverter,
                                                   &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> polang::createPolangToStandardPass() {
  return std::make_unique<PolangToStandardPass>();
}

void polang::registerPolangConversionPasses() {
  PassRegistration<PolangToStandardPass>();
}
