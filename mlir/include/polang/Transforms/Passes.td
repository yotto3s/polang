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

def NameResolution : Pass<"polang-resolve-names", "mlir::ModuleOp"> {
  let summary = "Resolve variable references to SSA values";
  let description = [{
    This pass resolves polang_ast.var_ref operations to their corresponding
    SSA values. It builds a scope table from function arguments and let
    bindings, then replaces var_ref ops with the appropriate SSA values.

    For variables referenced from outer scopes (captures), the pass:
    1. Detects the capture
    2. Adds the variable name to the function's captures attribute
    3. Emits an error for undefined variables
  }];

  let dependentDialects = ["polang::ast::PolangASTDialect"];
}

def TypeCheck : Pass<"polang-type-check", "mlir::ModuleOp"> {
  let summary = "Validate type correctness of AST dialect IR";
  let description = [{
    This pass validates semantic correctness after name resolution.
    It checks:
    - Binary operation types are compatible
    - If conditions are boolean
    - If branch types are compatible
    - Function call arity matches
    - Return types match function signatures

    Type variables are skipped (handled by type inference).
  }];

  let dependentDialects = ["polang::ast::PolangASTDialect"];
}

#endif // POLANG_TRANSFORMS_PASSES_TD
