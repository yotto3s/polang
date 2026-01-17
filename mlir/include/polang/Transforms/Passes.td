//===- Passes.td - Polang transformation passes ------------*- tablegen -*-===//
//
// Defines the transformation passes for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TRANSFORMS_PASSES_TD
#define POLANG_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TypeInference : Pass<"polang-type-inference", "mlir::ModuleOp"> {
  let summary = "Infer types for type variables using Hindley-Milner algorithm";
  let description = [{
    This pass performs Hindley-Milner style type inference on the Polang
    dialect. It resolves type variables (!polang.typevar<id>) to concrete
    types by collecting constraints from operations and solving them using
    unification.

    The pass:
    1. Collects type constraints from all operations
    2. Solves constraints using unification algorithm
    3. Substitutes resolved types for all type variables
    4. Reports errors for unsatisfiable constraints
  }];

  let dependentDialects = ["polang::PolangDialect"];
}

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
