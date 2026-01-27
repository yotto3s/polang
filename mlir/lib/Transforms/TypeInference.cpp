//===- TypeInference.cpp - Hindley-Milner type inference --------*- C++ -*-===//
//
// This file implements the type inference pass for the Polang dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"
#include "polang/Transforms/Passes.h"
#include "polang/Transforms/TypeInferenceUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#pragma GCC diagnostic pop

#include <optional>
#include <set>

using namespace mlir;
using namespace polang;

namespace {

/// Check if a function has any type variables in its signature.
/// The entry function (__polang_entry) is never considered polymorphic.
bool isPolymorphicFunction(FuncOp func) {
  // The entry function is never polymorphic - it should always be resolved
  if (func.getSymName() == "__polang_entry") {
    return false;
  }

  FunctionType funcType = func.getFunctionType();
  if (llvm::any_of(funcType.getInputs(),
                   [](Type input) { return containsTypeVar(input); })) {
    return true;
  }
  return llvm::any_of(funcType.getResults(),
                      [](Type result) { return containsTypeVar(result); });
}

//===----------------------------------------------------------------------===//
// TypeInferencePass - Main pass implementation
//===----------------------------------------------------------------------===//

struct TypeInferencePass
    : public PassWrapper<TypeInferencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeInferencePass)

  [[nodiscard]] StringRef getArgument() const override {
    return "polang-type-inference";
  }
  [[nodiscard]] StringRef getDescription() const override {
    return "Infer types for type variables using Hindley-Milner algorithm";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<PolangDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Substitution subst;
    Unifier unifier;
    bool hadError = false;

    // Phase 1: Collect constraints from function bodies
    module.walk([&](FuncOp func) {
      collectFunctionConstraints(func, subst, unifier, hadError);
    });

    if (hadError) {
      signalPassFailure();
      return;
    }

    // Phase 2: Collect constraints from call sites
    module.walk([&](CallOp call) {
      collectCallConstraints(call, module, subst, unifier, hadError);
    });

    if (hadError) {
      signalPassFailure();
      return;
    }

    // Phase 3: Apply substitution to resolve all types
    applySubstitution(module, subst, hadError);

    if (hadError) {
      signalPassFailure();
      return;
    }
  }

private:
  void collectFunctionConstraints(FuncOp func, Substitution& subst,
                                  Unifier& unifier, bool& hadError) {
    // For each return operation, unify return value type with function
    // return type
    FunctionType funcType = func.getFunctionType();
    if (funcType.getNumResults() == 0) {
      return;
    }

    Type expectedReturnType = funcType.getResult(0);

    func.walk([&](ReturnOp returnOp) {
      if (returnOp.getValue()) {
        Type actualType = returnOp.getValue().getType();
        if (!unifier.unify(expectedReturnType, actualType, subst)) {
          returnOp.emitError()
              << "return type mismatch: expected " << expectedReturnType
              << " but got " << actualType;
          hadError = true;
        }
      }
    });

    // For arithmetic operations, operands and result must have same type
    func.walk([&](Operation* op) {
      if (isa<AddOp, SubOp, MulOp, DivOp>(op)) {
        Type lhsType = op->getOperand(0).getType();
        Type rhsType = op->getOperand(1).getType();
        Type resultType = op->getResult(0).getType();

        if (!unifier.unify(lhsType, rhsType, subst)) {
          op->emitError() << "operand type mismatch: " << lhsType << " vs "
                          << rhsType;
          hadError = true;
        }
        if (!unifier.unify(lhsType, resultType, subst)) {
          op->emitError() << "result type mismatch: " << lhsType << " vs "
                          << resultType;
          hadError = true;
        }
      }
    });

    // For if operations, condition must be bool
    func.walk([&](IfOp ifOp) {
      Type condType = ifOp.getCondition().getType();
      Type boolType = BoolType::get(ifOp.getContext());
      if (!unifier.unify(condType, boolType, subst)) {
        ifOp.emitError() << "if condition must be bool, got " << condType;
        hadError = true;
      }
    });
  }

  void collectCallConstraints(CallOp call, ModuleOp module, Substitution& subst,
                              Unifier& unifier, bool& hadError) {
    // Find the callee function
    auto callee = module.lookupSymbol<FuncOp>(call.getCallee());
    if (!callee) {
      return;
    }

    // Skip polymorphic function calls - they can be called with different types
    // at different call sites (e.g., identity(42) and identity(true)).
    // Monomorphization handles creating specialized versions for each call
    // site.
    if (isPolymorphicFunction(callee)) {
      return;
    }

    FunctionType calleeType = callee.getFunctionType();

    // Unify argument types with parameter types
    for (size_t i = 0;
         i < call.getOperands().size() && i < calleeType.getNumInputs(); ++i) {
      Type argType = call.getOperand(i).getType();
      Type paramType = calleeType.getInput(i);

      if (!unifier.unify(argType, paramType, subst)) {
        call.emitError() << "argument type mismatch at position " << i << ": "
                         << argType << " vs " << paramType;
        hadError = true;
      }
    }

    // Unify call result type with callee return type
    if (call.getResult() && calleeType.getNumResults() > 0) {
      Type resultType = call.getResult().getType();
      Type returnType = calleeType.getResult(0);

      if (!unifier.unify(resultType, returnType, subst)) {
        call.emitError() << "return type mismatch: " << resultType << " vs "
                         << returnType;
        hadError = true;
      }
    }
  }

  void applySubstitution(ModuleOp module, const Substitution& subst,
                         [[maybe_unused]] bool& hadError) {
    // First, identify which functions are polymorphic (before any
    // modifications)
    llvm::StringSet<> polymorphicFuncs;
    module.walk([&](FuncOp func) {
      if (isPolymorphicFunction(func)) {
        polymorphicFuncs.insert(func.getSymName());
      }
    });

    // For CallOps to polymorphic functions, store the actual argument types
    // and compute the return type for the monomorphization pass.
    // This must happen BEFORE we update function signatures.
    module.walk([&](CallOp call) {
      if (polymorphicFuncs.contains(call.getCallee())) {
        // Get the callee function
        auto callee = module.lookupSymbol<FuncOp>(call.getCallee());
        if (!callee) {
          return;
        }

        FunctionType calleeType = callee.getFunctionType();

        // Store the actual argument types (they should already be concrete
        // or resolvable through the global substitution)
        SmallVector<Attribute> resolvedArgTypes;
        auto* ctx = call.getContext();
        for (Value arg : call.getOperands()) {
          Type resolvedType = subst.apply(arg.getType());
          // Apply defaults for numeric type vars (integer -> i64, float -> f64)
          resolvedType = applyTypeVarDefault(resolvedType, ctx);
          resolvedArgTypes.push_back(TypeAttr::get(resolvedType));
        }
        call->setAttr("polang.resolved_arg_types",
                      ArrayAttr::get(ctx, resolvedArgTypes));

        // Compute the return type by building a local substitution
        // that maps the function's parameter type vars to the argument types
        Substitution localSubst;

        // Map parameter type variables to argument types
        for (size_t i = 0;
             i < calleeType.getNumInputs() && i < call.getOperands().size();
             ++i) {
          Type paramType = calleeType.getInput(i);
          if (auto typeVar = dyn_cast<TypeVarType>(paramType)) {
            Type argType = subst.apply(call.getOperand(i).getType());
            localSubst.bind(typeVar.getId(), argType);
          }
        }

        // Also include the global substitution to handle relationships
        // established in the function body (e.g., return type = param type)
        Substitution combinedSubst = localSubst.compose(subst);

        // Apply to the return type and update the call's result type
        if (call.getResult()) {
          Type returnType = calleeType.getResult(0);
          Type resolvedRetType = combinedSubst.apply(returnType);
          // Apply defaults for numeric type vars (integer -> i64, float -> f64)
          resolvedRetType = applyTypeVarDefault(resolvedRetType, ctx);
          call->setAttr("polang.resolved_return_type",
                        TypeAttr::get(resolvedRetType));
          // Also update the actual result type so subsequent uses see the
          // concrete type
          if (!containsTypeVar(resolvedRetType)) {
            call.getResult().setType(resolvedRetType);
          }
        }
      }
    });

    // Update function signatures - but SKIP polymorphic functions
    // (they will be handled by the monomorphization pass)
    module.walk([&](FuncOp func) {
      // Skip polymorphic functions - preserve their type variables
      if (polymorphicFuncs.contains(func.getSymName())) {
        return;
      }

      FunctionType oldType = func.getFunctionType();
      auto* ctx = func.getContext();

      SmallVector<Type> newInputs;
      for (Type input : oldType.getInputs()) {
        Type resolved = subst.apply(input);
        // Apply defaults for numeric type vars (integer -> i64, float -> f64)
        resolved = applyTypeVarDefault(resolved, ctx);
        newInputs.push_back(resolved);
      }

      SmallVector<Type> newResults;
      for (Type result : oldType.getResults()) {
        Type resolved = subst.apply(result);
        // Apply defaults for numeric type vars (integer -> i64, float -> f64)
        resolved = applyTypeVarDefault(resolved, ctx);
        newResults.push_back(resolved);
      }

      FunctionType newType =
          FunctionType::get(func.getContext(), newInputs, newResults);

      if (newType != oldType) {
        func.setType(newType);

        // Update block argument types
        if (!func.getBody().empty()) {
          Block& entry = func.getBody().front();
          for (size_t i = 0; i < entry.getNumArguments(); ++i) {
            entry.getArgument(i).setType(newInputs[i]);
          }
        }
      }
    });

    // Now rebuild operations that have type variable results
    // We need to do this carefully to update all uses
    // Skip operations inside polymorphic functions
    module.walk([&](Operation* op) {
      // Skip function ops (handled above)
      if (isa<FuncOp>(op)) {
        return;
      }

      // Skip operations inside polymorphic functions
      if (auto parentFunc = op->getParentOfType<FuncOp>()) {
        if (polymorphicFuncs.contains(parentFunc.getSymName())) {
          return;
        }
      }

      auto* ctx = op->getContext();
      bool needsUpdate = false;
      for (Type type : op->getResultTypes()) {
        Type resolved = subst.apply(type);
        // Apply defaults for numeric type vars
        resolved = applyTypeVarDefault(resolved, ctx);
        if (resolved != type) {
          needsUpdate = true;
          break;
        }
      }

      if (!needsUpdate) {
        return;
      }

      // Update result types by creating new operation
      OpBuilder builder(op);
      SmallVector<Type> newResultTypes;
      for (Type type : op->getResultTypes()) {
        Type resolved = subst.apply(type);
        // Apply defaults for numeric type vars (integer -> i64, float -> f64)
        resolved = applyTypeVarDefault(resolved, ctx);
        newResultTypes.push_back(resolved);
      }

      // For most operations, we can just change the result type
      // This requires rebuilding the operation
      OperationState state(op->getLoc(), op->getName());
      state.addOperands(op->getOperands());
      state.addTypes(newResultTypes);
      state.addAttributes(op->getAttrs());

      // Copy regions
      for (Region& region : op->getRegions()) {
        Region* newRegion = state.addRegion();
        IRMapping mapping;
        region.cloneInto(newRegion, mapping);
      }

      Operation* newOp = builder.create(state);

      // Replace uses of old results with new results
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
      }

      op->erase();
    });
  }
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createTypeInferencePass() {
  return std::make_unique<TypeInferencePass>();
}

} // namespace polang
