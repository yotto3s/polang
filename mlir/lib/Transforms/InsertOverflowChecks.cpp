//===- InsertOverflowChecks.cpp - Insert overflow checks --------*- C++ -*-===//
//
// This file implements the pass that inserts runtime overflow checks for
// integer arithmetic operations.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#pragma GCC diagnostic pop

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Get or create the runtime error handler function declaration
func::FuncOp getOrCreateRuntimeErrorHandler(ModuleOp module,
                                            PatternRewriter& rewriter) {
  // Check if function already exists
  if (auto existingFunc =
          module.lookupSymbol<func::FuncOp>("__polang_runtime_error")) {
    return existingFunc;
  }

  // Create function type: (ptr<i8>, i32, i32) -> void
  // Arguments: message (string pointer), line, column
  auto i8PtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i32Type = rewriter.getI32Type();
  auto funcType = rewriter.getFunctionType(
      {i8PtrType, i32Type, i32Type}, {});

  // Create function declaration
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  auto funcOp = rewriter.create<func::FuncOp>(
      module.getLoc(), "__polang_runtime_error", funcType);
  funcOp.setPrivate();

  return funcOp;
}

/// Get or create a global string constant for the error message
LLVM::GlobalOp getOrCreateErrorMessage(ModuleOp module, StringRef message,
                                       PatternRewriter& rewriter) {
  // Create a unique name for this message
  std::string globalName = "__polang_error_msg_" + message.str();
  
  // Check if it already exists
  if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
    return existing;
  }

  // Create the global string
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  auto loc = module.getLoc();
  auto i8Type = rewriter.getI8Type();
  auto arrayType = LLVM::LLVMArrayType::get(i8Type, message.size() + 1);
  
  auto globalOp = rewriter.create<LLVM::GlobalOp>(
      loc, arrayType, /*isConstant=*/true, LLVM::Linkage::Private,
      globalName, rewriter.getStringAttr(message.str() + '\0'));

  return globalOp;
}

//===----------------------------------------------------------------------===//
// Overflow check patterns
//===----------------------------------------------------------------------===//

/// Base pattern for inserting overflow checks
template <typename ArithOp, typename SignedIntrinsic, typename UnsignedIntrinsic>
struct InsertOverflowCheckPattern : public OpRewritePattern<ArithOp> {
  using OpRewritePattern<ArithOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArithOp op,
                                PatternRewriter& rewriter) const override {
    // Only process integer operations
    auto resultType = dyn_cast<IntegerType>(op.getType());
    if (!resultType) {
      return failure();
    }

    // Check if this op has the signedness attribute
    auto isUnsignedAttr = op->template getAttrOfType<BoolAttr>("polang.is_unsigned");
    bool isUnsigned = isUnsignedAttr && isUnsignedAttr.getValue();

    auto loc = op.getLoc();
    auto module = op->template getParentOfType<ModuleOp>();
    
    // Get the overflow intrinsic based on signedness and width
    unsigned width = resultType.getWidth();
    auto i1Type = rewriter.getI1Type();
    auto overflowResultType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {resultType, i1Type});

    // Create the overflow intrinsic
    Value overflowOp;
    if (isUnsigned) {
      std::string intrinsicName = "llvm.uadd.with.overflow.i" + std::to_string(width);
      if (std::is_same<ArithOp, SubIOp>::value) {
        intrinsicName = "llvm.usub.with.overflow.i" + std::to_string(width);
      } else if (std::is_same<ArithOp, MulIOp>::value) {
        intrinsicName = "llvm.umul.with.overflow.i" + std::to_string(width);
      }
      
      overflowOp = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, overflowResultType, intrinsicName,
          ValueRange{op.getLhs(), op.getRhs()});
    } else {
      std::string intrinsicName = "llvm.sadd.with.overflow.i" + std::to_string(width);
      if (std::is_same<ArithOp, SubIOp>::value) {
        intrinsicName = "llvm.ssub.with.overflow.i" + std::to_string(width);
      } else if (std::is_same<ArithOp, MulIOp>::value) {
        intrinsicName = "llvm.smul.with.overflow.i" + std::to_string(width);
      }
      
      overflowOp = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, overflowResultType, intrinsicName,
          ValueRange{op.getLhs(), op.getRhs()});
    }

    // Extract result and overflow flag
    auto result = rewriter.create<LLVM::ExtractValueOp>(
        loc, resultType, overflowOp, ArrayRef<int64_t>{0});
    auto overflowFlag = rewriter.create<LLVM::ExtractValueOp>(
        loc, i1Type, overflowOp, ArrayRef<int64_t>{1});

    // Create blocks for overflow and no-overflow paths
    Block* currentBlock = rewriter.getInsertionBlock();
    Block* overflowBlock = rewriter.splitBlock(currentBlock,
                                               rewriter.getInsertionPoint());
    Block* continueBlock = rewriter.splitBlock(overflowBlock,
                                              overflowBlock->begin());

    // Insert conditional branch
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cf::CondBranchOp>(loc, overflowFlag, overflowBlock,
                                     ArrayRef<Value>{}, continueBlock,
                                     ArrayRef<Value>{});

    // Fill in the overflow block - call error handler and exit
    rewriter.setInsertionPointToStart(overflowBlock);
    
    // Get or create the error handler
    auto errorHandler = getOrCreateRuntimeErrorHandler(module, rewriter);
    
    // Create error message
    std::string message = "integer overflow";
    auto errorMsg = getOrCreateErrorMessage(module, message, rewriter);
    
    // Get pointer to the message
    auto i8PtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto msgPtr = rewriter.create<LLVM::GEPOp>(
        loc, i8PtrType, errorMsg.getType(), errorMsg,
        ArrayRef<LLVM::GEPArg>{0, 0}, /*inbounds=*/true);

    // Extract line and column from location (placeholder for now)
    auto line = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto column = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    // Call error handler
    rewriter.create<func::CallOp>(loc, errorHandler,
                                 ValueRange{msgPtr, line, column});
    
    // Unreachable after error
    rewriter.create<LLVM::UnreachableOp>(loc);

    // Continue block - replace original op with the result
    rewriter.setInsertionPointToStart(continueBlock);
    rewriter.replaceOp(op, result.getResult());

    return success();
  }
};

using InsertAddOverflowCheck =
    InsertOverflowCheckPattern<AddIOp, void, void>;
using InsertSubOverflowCheck =
    InsertOverflowCheckPattern<SubIOp, void, void>;
using InsertMulOverflowCheck =
    InsertOverflowCheckPattern<MulIOp, void, void>;

//===----------------------------------------------------------------------===//
// InsertOverflowChecksPass
//===----------------------------------------------------------------------===//

struct InsertOverflowChecksPass
    : public PassWrapper<InsertOverflowChecksPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertOverflowChecksPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "polang-insert-overflow-checks";
  }

  [[nodiscard]] StringRef getDescription() const override {
    return "Insert runtime overflow checks for integer arithmetic";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    cf::ControlFlowDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    
    // Apply overflow check patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<InsertAddOverflowCheck, InsertSubOverflowCheck,
                 InsertMulOverflowCheck>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createInsertOverflowChecksPass() {
  return std::make_unique<InsertOverflowChecksPass>();
}

} // namespace polang
