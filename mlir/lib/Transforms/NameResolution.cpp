//===- NameResolution.cpp - Resolve variable references ---------*- C++ -*-===//
//
// This file implements the name resolution pass for the Polang AST dialect.
// It resolves polang_ast.var_ref operations to their corresponding SSA values.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangASTOps.h"
#include "polang/Dialect/PolangTypes.h"
#include "polang/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"

#pragma GCC diagnostic pop

using namespace mlir;

// Aliases for AST dialect types to avoid conflicts with mlir:: types
using ASTFuncOp = polang::ast::FuncOp;
using ASTLetExprOp = polang::ast::LetExprOp;
using ASTVarRefOp = polang::ast::VarRefOp;

namespace {

//===----------------------------------------------------------------------===//
// NameResolutionPass
//===----------------------------------------------------------------------===//

struct NameResolutionPass
    : public PassWrapper<NameResolutionPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NameResolutionPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "polang-resolve-names";
  }

  [[nodiscard]] StringRef getDescription() const override {
    return "Resolve variable references to SSA values";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<polang::ast::PolangASTDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    bool hadError = false;

    // Process each function in the module
    module.walk([&](ASTFuncOp func) { resolveFunction(func, hadError); });

    if (hadError) {
      signalPassFailure();
    }
  }

private:
  /// Find the value for a variable name by walking up the scope chain
  Value lookupVariable(StringRef name, Operation* fromOp, ASTFuncOp func) {
    // Walk up from the operation to find scopes
    Operation* current = fromOp;
    while (current && current != func.getOperation()) {
      // Check if we're in a let expression body
      if (auto letExpr = dyn_cast<ASTLetExprOp>(current)) {
        // Check if name matches any binding
        auto varNames = letExpr.getVarNames();
        Block& bodyBlock = letExpr.getBody().front();
        for (size_t i = 0; i < varNames.size(); ++i) {
          auto bindingName = cast<StringAttr>(varNames[i]).getValue();
          if (bindingName == name) {
            return bodyBlock.getArgument(i);
          }
        }
      }
      current = current->getParentOp();
    }

    // Check function arguments
    // Function argument names are stored in arg_attrs with "polang.name" key
    Block& entry = func.getBody().front();
    if (auto argAttrs = func.getArgAttrsAttr()) {
      for (size_t i = 0; i < argAttrs.size(); ++i) {
        auto dict = cast<DictionaryAttr>(argAttrs[i]);
        if (auto nameAttr = dict.getAs<StringAttr>("polang.name")) {
          if (nameAttr.getValue() == name) {
            return entry.getArgument(i);
          }
        }
      }
    }

    // Not found in any scope
    return nullptr;
  }

  /// Resolve all variable references within a function
  void resolveFunction(ASTFuncOp func, bool& hadError) {
    // Collect all var_refs to resolve (can't modify while walking)
    SmallVector<ASTVarRefOp> varRefsToResolve;
    func.walk([&](ASTVarRefOp varRef) { varRefsToResolve.push_back(varRef); });

    // Resolve each var_ref
    for (ASTVarRefOp varRef : varRefsToResolve) {
      StringRef varName = varRef.getVarName();

      // Try to resolve the variable
      Value resolved = lookupVariable(varName, varRef.getOperation(), func);

      if (resolved) {
        // Replace all uses of the var_ref with the resolved value
        varRef.getResult().replaceAllUsesWith(resolved);
        varRef.erase();
      } else {
        // Undefined variable
        varRef.emitError() << "undefined variable '" << varName << "'";
        hadError = true;
      }
    }
  }
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createNameResolutionPass() {
  return std::make_unique<NameResolutionPass>();
}

} // namespace polang
