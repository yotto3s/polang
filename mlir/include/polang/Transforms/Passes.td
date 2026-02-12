//===- Passes.td - Polang transformation passes ------------*- tablegen -*-===//
//
// Defines the transformation passes for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TRANSFORMS_PASSES_TD
#define POLANG_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def Monomorphization : Pass<"polang-monomorphize", "mlir::ModuleOp"> {
  let summary = "Monomorphize polymorphic functions";
  let description = [{
    This pass specializes polymorphic functions for each unique set of
    concrete type arguments used at call sites. After this pass, all
    functions have concrete types and can be lowered to LLVM.
  }];

  let dependentDialects = ["polang::PolangDialect"];
}

def InsertOverflowChecks : Pass<"polang-insert-overflow-checks", "mlir::ModuleOp"> {
  let summary = "Insert runtime overflow checks for integer arithmetic";
  let description = [{
    This pass converts integer arithmetic operations (add, sub, mul) to
    LLVM overflow intrinsics with runtime checks. When overflow is detected,
    the program prints an error message with source location and exits.

    This pass should run after all MLIR optimizations but before the final
    ArithToLLVM conversion pass.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

#endif // POLANG_TRANSFORMS_PASSES_TD
