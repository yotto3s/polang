//===- ASTToPolang.cpp - AST to Polang dialect conversion --------*- C++ -*-===//
//
// This file implements the conversion pass from the Polang AST dialect to the
// Polang dialect. This pass handles:
// - Converting AST operations to their Polang equivalents
// - Inlining let expressions (AST let_expr becomes SSA values)
// - Preserving type variables for later type inference
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Conversion/Passes.h"
#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangASTOps.h"
#include "polang/Dialect/PolangASTTypes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#pragma GCC diagnostic pop

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

/// Convert AST dialect types to Polang dialect types.
/// Most types are shared between dialects; AST-specific types need conversion.
class ASTToPolangTypeConverter : public TypeConverter {
public:
  ASTToPolangTypeConverter() {
    // Default: keep types unchanged (IntegerType, FloatType, BoolType are
    // shared)
    addConversion([](Type type) { return type; });

    // Convert AST TypeVarType to Polang TypeVarType
    // Note: Both dialects share the same TypeVarType from the Polang dialect
    addConversion([](polang::ast::TypeVarType type) -> Type {
      // AST TypeVarType and Polang TypeVarType are the same type
      // The AST dialect imports TypeVarType from the Polang dialect
      return type;
    });
  }
};

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

struct ConstantIntegerOpConversion
    : public OpConversionPattern<polang::ast::ConstantIntegerOp> {
  using OpConversionPattern<polang::ast::ConstantIntegerOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::ConstantIntegerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<polang::ConstantIntegerOp>(
        op, resultType, op.getValueAttr());
    return success();
  }
};

struct ConstantFloatOpConversion
    : public OpConversionPattern<polang::ast::ConstantFloatOp> {
  using OpConversionPattern<polang::ast::ConstantFloatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::ConstantFloatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<polang::ConstantFloatOp>(
        op, resultType, op.getValueAttr());
    return success();
  }
};

struct ConstantBoolOpConversion
    : public OpConversionPattern<polang::ast::ConstantBoolOp> {
  using OpConversionPattern<polang::ast::ConstantBoolOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::ConstantBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    // ConstantBoolOp result type is always !polang.bool
    rewriter.replaceOpWithNewOp<polang::ConstantBoolOp>(
        op, rewriter.getBoolAttr(op.getValue()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

struct AddOpConversion : public OpConversionPattern<polang::ast::AddOp> {
  using OpConversionPattern<polang::ast::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<polang::AddOp>(op, resultType, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

struct SubOpConversion : public OpConversionPattern<polang::ast::SubOp> {
  using OpConversionPattern<polang::ast::SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<polang::SubOp>(op, resultType, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

struct MulOpConversion : public OpConversionPattern<polang::ast::MulOp> {
  using OpConversionPattern<polang::ast::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<polang::MulOp>(op, resultType, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

struct DivOpConversion : public OpConversionPattern<polang::ast::DivOp> {
  using OpConversionPattern<polang::ast::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<polang::DivOp>(op, resultType, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

struct CmpOpConversion : public OpConversionPattern<polang::ast::CmpOp> {
  using OpConversionPattern<polang::ast::CmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<polang::CmpOp>(
        op, polang::BoolType::get(op.getContext()), op.getPredicate(),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

struct FuncOpConversion : public OpConversionPattern<polang::ast::FuncOp> {
  using OpConversionPattern<polang::ast::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    auto funcType = op.getFunctionType();
    const auto* typeConverter = getTypeConverter();

    // Convert argument types
    TypeConverter::SignatureConversion signatureConversion(
        funcType.getNumInputs());
    for (size_t i = 0; i < funcType.getNumInputs(); ++i) {
      signatureConversion.addInputs(
          i, typeConverter->convertType(funcType.getInput(i)));
    }

    // Convert result types
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(funcType.getResults(), resultTypes))) {
      return failure();
    }

    auto newFuncType = rewriter.getFunctionType(
        signatureConversion.getConvertedTypes(), resultTypes);

    // Create the new Polang function
    auto newFunc = rewriter.create<polang::FuncOp>(op.getLoc(), op.getSymName(),
                                                    newFuncType);

    // Copy arg_attrs if present
    if (auto argAttrs = op.getArgAttrs()) {
      newFunc.setArgAttrsAttr(*argAttrs);
    }

    // Move the function body
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<polang::ast::CallOp> {
  using OpConversionPattern<polang::ast::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<polang::CallOp>(op, op.getCallee(), resultTypes,
                                                 adaptor.getOperands());
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<polang::ast::ReturnOp> {
  using OpConversionPattern<polang::ast::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // ReturnOp takes an optional single value
    Value returnValue = adaptor.getValue();
    rewriter.replaceOpWithNewOp<polang::ReturnOp>(op, returnValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

struct IfOpConversion : public OpConversionPattern<polang::ast::IfOp> {
  using OpConversionPattern<polang::ast::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }

    // Create the new Polang if operation
    auto newIf = rewriter.create<polang::IfOp>(op.getLoc(), resultType,
                                                adaptor.getCondition());

    // Move then region
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().end());
    // Move else region
    rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(),
                                newIf.getElseRegion().end());

    rewriter.replaceOp(op, newIf.getResult());
    return success();
  }
};

struct YieldOpConversion : public OpConversionPattern<polang::ast::YieldOp> {
  using OpConversionPattern<polang::ast::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // YieldOp takes a single value
    Value yieldValue = adaptor.getValue();
    rewriter.replaceOpWithNewOp<polang::YieldOp>(op, yieldValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Let Expression Inlining
//===----------------------------------------------------------------------===//

/// Let expressions are inlined - the binding values replace the block arguments
/// in the body, and the body's yield value becomes the result.
struct LetExprOpConversion
    : public OpConversionPattern<polang::ast::LetExprOp> {
  using OpConversionPattern<polang::ast::LetExprOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(polang::ast::LetExprOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    // Collect the binding values from the binding regions
    SmallVector<Value> bindingValues;
    for (Region& bindingRegion : op.getBindings()) {
      // Each binding region has a single block with a yield.binding terminator
      Block& block = bindingRegion.front();
      auto yieldBinding =
          cast<polang::ast::YieldBindingOp>(block.getTerminator());
      bindingValues.push_back(yieldBinding.getValue());
    }

    // Get the body block and its arguments
    Region& bodyRegion = op.getBody();
    Block& bodyBlock = bodyRegion.front();

    // Map body block arguments to binding values
    IRMapping mapping;
    for (auto [arg, val] :
         llvm::zip(bodyBlock.getArguments(), bindingValues)) {
      mapping.map(arg, val);
    }

    // Clone the body operations (except the terminator) before the let_expr
    rewriter.setInsertionPoint(op);
    for (Operation& bodyOp : bodyBlock.without_terminator()) {
      rewriter.clone(bodyOp, mapping);
    }

    // Get the yield value from the body
    auto yield = cast<polang::ast::YieldOp>(bodyBlock.getTerminator());
    Value resultValue = mapping.lookupOrDefault(yield.getValue());

    // Replace the let_expr with the yield value
    rewriter.replaceOp(op, resultValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Module Operations
//===----------------------------------------------------------------------===//

/// AST ModuleOp converts to the standard mlir::ModuleOp by just keeping it
/// since both dialects use mlir::ModuleOp as the top-level container.
/// The conversion framework handles module preservation automatically.

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ASTToPolangPass
    : public PassWrapper<ASTToPolangPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ASTToPolangPass)

  [[nodiscard]] StringRef getArgument() const final {
    return "polang-ast-to-polang";
  }

  [[nodiscard]] StringRef getDescription() const final {
    return "Convert AST dialect to Polang dialect";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<polang::ast::PolangASTDialect, polang::PolangDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // Polang dialect is legal
    target.addLegalDialect<polang::PolangDialect>();

    // AST dialect is illegal (must be converted)
    target.addIllegalDialect<polang::ast::PolangASTDialect>();

    ASTToPolangTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());

    patterns.add<ConstantIntegerOpConversion, ConstantFloatOpConversion,
                 ConstantBoolOpConversion, AddOpConversion, SubOpConversion,
                 MulOpConversion, DivOpConversion, CmpOpConversion,
                 FuncOpConversion, CallOpConversion, ReturnOpConversion,
                 IfOpConversion, YieldOpConversion, LetExprOpConversion>(
        typeConverter, &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace polang {

std::unique_ptr<mlir::Pass> createASTToPolangPass() {
  return std::make_unique<ASTToPolangPass>();
}

} // namespace polang
