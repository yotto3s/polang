//===- InsertOverflowChecks.cpp - Insert overflow checks --------*- C++ -*-===//
//
// This file implements passes for integer overflow checking:
// - CheckConstantOverflowPass: compile-time detection of constant-expression
//   overflow (runs before canonicalization to catch cases that would be folded)
// - InsertOverflowChecksPass: runtime overflow checks via LLVM intrinsics
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APInt.h"

#pragma GCC diagnostic pop

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// CheckConstantOverflowPass
//===----------------------------------------------------------------------===//

/// Check whether a constant integer arithmetic operation overflows.
/// Returns true if overflow is detected.
bool checkConstantArithOverflow(Operation* op, const llvm::APInt& lhs,
                                const llvm::APInt& rhs, bool isUnsigned) {
  bool overflow = false;
  const auto opName = op->getName().getStringRef();

  if (opName == "arith.addi") {
    if (isUnsigned) {
      (void)lhs.uadd_ov(rhs, overflow);
    } else {
      (void)lhs.sadd_ov(rhs, overflow);
    }
  } else if (opName == "arith.subi") {
    if (isUnsigned) {
      (void)lhs.usub_ov(rhs, overflow);
    } else {
      (void)lhs.ssub_ov(rhs, overflow);
    }
  } else if (opName == "arith.muli") {
    if (isUnsigned) {
      (void)lhs.umul_ov(rhs, overflow);
    } else {
      (void)lhs.smul_ov(rhs, overflow);
    }
  }

  return overflow;
}

/// Get the APInt value from an arith.constant operation, if possible.
std::optional<llvm::APInt> getConstantIntValue(Value value) {
  auto defOp = value.getDefiningOp<arith::ConstantOp>();
  if (!defOp) {
    return std::nullopt;
  }
  auto intAttr = dyn_cast<IntegerAttr>(defOp.getValue());
  if (!intAttr) {
    return std::nullopt;
  }
  return intAttr.getValue();
}

/// Compile-time overflow check pass.
/// Detects constant-expression overflow BEFORE canonicalization can fold it.
struct CheckConstantOverflowPass
    : public PassWrapper<CheckConstantOverflowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckConstantOverflowPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "polang-check-constant-overflow";
  }

  [[nodiscard]] StringRef getDescription() const override {
    return "Check for overflow in constant integer arithmetic expressions";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    bool hasError = false;

    module.walk([&](Operation* op) {
      if (!isa<AddIOp, SubIOp, MulIOp>(op)) {
        return;
      }

      auto resultType = dyn_cast<IntegerType>(op->getResult(0).getType());
      if (!resultType) {
        return;
      }

      // Only check operations where both operands are constants
      auto lhsVal = getConstantIntValue(op->getOperand(0));
      auto rhsVal = getConstantIntValue(op->getOperand(1));
      if (!lhsVal || !rhsVal) {
        return;
      }

      const auto isUnsignedAttr =
          op->getAttrOfType<BoolAttr>("polang.is_unsigned");
      const bool isUnsigned = isUnsignedAttr && isUnsignedAttr.getValue();

      if (checkConstantArithOverflow(op, *lhsVal, *rhsVal, isUnsigned)) {
        op->emitError("integer overflow in constant expression");
        hasError = true;
      }
    });

    if (hasError) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Get or create the runtime error handler function declaration.
LLVMFuncOp getOrCreateRuntimeErrorHandler(ModuleOp module, OpBuilder& builder) {
  // Check for existing LLVM function declaration (may already exist if
  // DivOpLowering created a func::FuncOp that FuncToLLVM converted)
  if (auto existingFunc =
          module.lookupSymbol<LLVMFuncOp>("__polang_runtime_error")) {
    return existingFunc;
  }

  const auto i8PtrType = LLVMPointerType::get(builder.getContext());
  const auto i32Type = builder.getI32Type();
  const auto voidType = LLVMVoidType::get(builder.getContext());
  const auto funcType =
      LLVMFunctionType::get(voidType, {i8PtrType, i32Type, i32Type});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto funcOp = builder.create<LLVMFuncOp>(module.getLoc(),
                                           "__polang_runtime_error", funcType);
  funcOp.setLinkage(Linkage::External);

  return funcOp;
}

/// Get or create a global string constant for the error message.
GlobalOp getOrCreateErrorMessage(ModuleOp module, StringRef message,
                                 OpBuilder& builder) {
  // Create a name with no spaces (replace spaces with underscores)
  std::string globalName = "__polang_error_msg_";
  for (const char c : message) {
    globalName += (c == ' ') ? '_' : c;
  }

  if (auto existing = module.lookupSymbol<GlobalOp>(globalName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  const auto loc = module.getLoc();
  const auto i8Type = builder.getI8Type();
  const auto arrayType = LLVMArrayType::get(i8Type, message.size() + 1);

  auto globalOp = builder.create<GlobalOp>(
      loc, arrayType, /*isConstant=*/true, Linkage::Private, globalName,
      builder.getStringAttr(message.str() + '\0'));

  return globalOp;
}

/// Determine the overflow intrinsic name for the given operation.
[[nodiscard]] std::string
getOverflowIntrinsicName(Operation* op, bool isUnsigned, unsigned width) {
  const auto opName = op->getName().getStringRef();

  std::string prefix;
  if (opName == "arith.addi") {
    prefix = isUnsigned ? "llvm.uadd" : "llvm.sadd";
  } else if (opName == "arith.subi") {
    prefix = isUnsigned ? "llvm.usub" : "llvm.ssub";
  } else if (opName == "arith.muli") {
    prefix = isUnsigned ? "llvm.umul" : "llvm.smul";
  }

  return prefix + ".with.overflow.i" + std::to_string(width);
}

/// Insert an overflow check for a single arithmetic operation.
/// Replaces the operation with an overflow intrinsic call, branching to
/// an error handler on overflow.
LogicalResult insertOverflowCheck(Operation* op, OpBuilder& builder,
                                  ModuleOp module) {
  // Only process integer operations
  // TODO(#76): IndexType overflow checking requires converting to
  // fixed-width integer first. Defer to a follow-up PR.
  const auto resultType = dyn_cast<IntegerType>(op->getResult(0).getType());
  if (!resultType) {
    return success();
  }

  // Check signedness attribute
  const auto isUnsignedAttr = op->getAttrOfType<BoolAttr>("polang.is_unsigned");
  const bool isUnsigned = isUnsignedAttr && isUnsignedAttr.getValue();

  const auto loc = op->getLoc();
  const unsigned width = resultType.getWidth();
  const auto i1Type = builder.getI1Type();
  const auto overflowResultType =
      LLVMStructType::getLiteral(builder.getContext(), {resultType, i1Type});

  // Determine intrinsic name
  const std::string intrinsicName =
      getOverflowIntrinsicName(op, isUnsigned, width);

  // Get operands from the original op
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);

  // Insert the overflow intrinsic BEFORE the original op
  builder.setInsertionPoint(op);

  auto callOp = builder.create<CallIntrinsicOp>(
      loc, overflowResultType, builder.getStringAttr(intrinsicName),
      ValueRange{lhs, rhs});
  Value overflowResult = callOp.getResult(0);

  // Extract the computed value and overflow flag
  auto result = builder.create<ExtractValueOp>(loc, resultType, overflowResult,
                                               ArrayRef<int64_t>{0});
  auto overflowFlag = builder.create<ExtractValueOp>(
      loc, i1Type, overflowResult, ArrayRef<int64_t>{1});

  // Split the block: op and everything after it goes to continueBlock
  Block* currentBlock = op->getBlock();
  Block* continueBlock = currentBlock->splitBlock(op);

  // Create overflow error block between currentBlock and continueBlock
  auto* overflowBlock = new Block();
  currentBlock->getParent()->getBlocks().insertAfter(
      currentBlock->getIterator(), overflowBlock);

  // Terminate currentBlock with conditional branch
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<cf::CondBranchOp>(loc, overflowFlag, overflowBlock,
                                   ValueRange{}, continueBlock, ValueRange{});

  // Fill in the overflow error block
  builder.setInsertionPointToStart(overflowBlock);

  auto errorHandler = getOrCreateRuntimeErrorHandler(module, builder);

  const std::string message = "integer overflow";
  auto errorMsg = getOrCreateErrorMessage(module, message, builder);

  const auto i8PtrType = LLVMPointerType::get(builder.getContext());
  auto msgPtr =
      builder.create<AddressOfOp>(loc, i8PtrType, errorMsg.getSymName());

  // Extract source location for meaningful error messages
  int32_t lineNum = 0;
  int32_t colNum = 0;
  if (auto fileLoc = dyn_cast<FileLineColLoc>(op->getLoc())) {
    lineNum = static_cast<int32_t>(fileLoc.getLine());
    colNum = static_cast<int32_t>(fileLoc.getColumn());
  } else if (auto fusedLoc = dyn_cast<FusedLoc>(op->getLoc())) {
    for (auto innerLoc : fusedLoc.getLocations()) {
      if (auto fileLoc = dyn_cast<FileLineColLoc>(innerLoc)) {
        lineNum = static_cast<int32_t>(fileLoc.getLine());
        colNum = static_cast<int32_t>(fileLoc.getColumn());
        break;
      }
    }
  }

  auto line = builder.create<LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(lineNum));
  auto column = builder.create<LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(colNum));

  builder.create<CallOp>(loc, errorHandler, ValueRange{msgPtr, line, column});
  builder.create<UnreachableOp>(loc);

  // Replace all uses of the original op's result with the extracted value,
  // then erase the original op
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();

  return success();
}

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
                    cf::ControlFlowDialect, LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Collect ops first to avoid modifying the IR while iterating
    SmallVector<Operation*> opsToRewrite;
    module.walk([&](Operation* op) {
      if (isa<AddIOp, SubIOp, MulIOp>(op)) {
        if (isa<IntegerType>(op->getResult(0).getType())) {
          opsToRewrite.push_back(op);
        }
      }
    });

    // TODO: Division overflow (MIN_INT / -1) requires an explicit
    // comparison, not an overflow intrinsic. Defer to the division
    // safety check pass.

    OpBuilder builder(&getContext());
    for (auto* op : opsToRewrite) {
      if (failed(insertOverflowCheck(op, builder, module))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createCheckConstantOverflowPass() {
  return std::make_unique<CheckConstantOverflowPass>();
}

std::unique_ptr<Pass> createInsertOverflowChecksPass() {
  return std::make_unique<InsertOverflowChecksPass>();
}

} // namespace polang
