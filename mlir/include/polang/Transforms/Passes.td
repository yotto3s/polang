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

#endif // POLANG_TRANSFORMS_PASSES_TD
