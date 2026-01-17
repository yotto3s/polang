//===- TypeInference.cpp - Hindley-Milner type inference --------*- C++ -*-===//
//
// This file implements the type inference pass for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#include "polang/Transforms/Passes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <set>

using namespace mlir;
using namespace polang;

namespace {

//===----------------------------------------------------------------------===//
// Substitution - Maps type variables to types
//===----------------------------------------------------------------------===//

class Substitution {
public:
  void bind(uint64_t var, Type type) { bindings_[var] = type; }

  Type lookup(uint64_t var) const {
    auto it = bindings_.find(var);
    return it != bindings_.end() ? it->second : Type();
  }

  bool contains(uint64_t var) const { return bindings_.count(var) > 0; }

  /// Apply substitution to a type, recursively resolving type variables
  Type apply(Type type) const {
    if (auto typeVar = dyn_cast<TypeVarType>(type)) {
      Type bound = lookup(typeVar.getId());
      if (bound) {
        // Recursively apply in case bound type contains more type vars
        return apply(bound);
      }
      return type;
    }
    // For now, only type variables can be substituted
    // TODO: Handle function types when added
    return type;
  }

  /// Compose two substitutions: (this . other)(t) = this(other(t))
  Substitution compose(const Substitution& other) const {
    Substitution result;
    // Apply this substitution to all types in other
    for (const auto& [var, type] : other.bindings_) {
      result.bindings_[var] = apply(type);
    }
    // Add bindings from this that aren't in other
    for (const auto& [var, type] : bindings_) {
      if (!result.contains(var)) {
        result.bindings_[var] = type;
      }
    }
    return result;
  }

  void dump() const {
    for (const auto& [var, type] : bindings_) {
      llvm::errs() << "  typevar<" << var << "> = " << type << "\n";
    }
  }

private:
  llvm::DenseMap<uint64_t, Type> bindings_;
};

//===----------------------------------------------------------------------===//
// Unifier - Implements the unification algorithm
//===----------------------------------------------------------------------===//

class Unifier {
public:
  /// Unify two types, returning true on success
  bool unify(Type t1, Type t2, Substitution& subst) {
    // Apply current substitution first
    Type s1 = subst.apply(t1);
    Type s2 = subst.apply(t2);

    // Same type - trivially unifiable
    if (s1 == s2)
      return true;

    // Left is type variable
    if (auto var1 = dyn_cast<TypeVarType>(s1)) {
      return unifyVar(var1.getId(), s2, subst);
    }

    // Right is type variable
    if (auto var2 = dyn_cast<TypeVarType>(s2)) {
      return unifyVar(var2.getId(), s1, subst);
    }

    // Both are concrete types but different - cannot unify
    return false;
  }

private:
  /// Check if type variable occurs in type (prevents infinite types)
  bool occursIn(uint64_t var, Type type) const {
    if (auto typeVar = dyn_cast<TypeVarType>(type)) {
      return typeVar.getId() == var;
    }
    // TODO: Check inside function types when added
    return false;
  }

  /// Unify a type variable with a type
  bool unifyVar(uint64_t var, Type type, Substitution& subst) {
    // Occurs check - prevent infinite types
    if (occursIn(var, type)) {
      return false;
    }
    subst.bind(var, type);
    return true;
  }
};

//===----------------------------------------------------------------------===//
// TypeInferencePass - Main pass implementation
//===----------------------------------------------------------------------===//

struct TypeInferencePass
    : public PassWrapper<TypeInferencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeInferencePass)

  StringRef getArgument() const override { return "polang-type-inference"; }
  StringRef getDescription() const override {
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
    applySubstitution(module, subst);
  }

private:
  void collectFunctionConstraints(FuncOp func, Substitution& subst,
                                   Unifier& unifier, bool& hadError) {
    // For each return operation, unify return value type with function
    // return type
    FunctionType funcType = func.getFunctionType();
    if (funcType.getNumResults() == 0)
      return;

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
  }

  void collectCallConstraints(CallOp call, ModuleOp module, Substitution& subst,
                               Unifier& unifier, bool& hadError) {
    // Find the callee function
    auto callee = module.lookupSymbol<FuncOp>(call.getCallee());
    if (!callee)
      return;

    FunctionType calleeType = callee.getFunctionType();

    // Unify argument types with parameter types
    for (size_t i = 0; i < call.getOperands().size() &&
                       i < calleeType.getNumInputs(); ++i) {
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

  void applySubstitution(ModuleOp module, const Substitution& subst) {
    // First update function signatures (this is easier)
    module.walk([&](FuncOp func) {
      FunctionType oldType = func.getFunctionType();

      SmallVector<Type> newInputs;
      for (Type input : oldType.getInputs()) {
        newInputs.push_back(subst.apply(input));
      }

      SmallVector<Type> newResults;
      for (Type result : oldType.getResults()) {
        newResults.push_back(subst.apply(result));
      }

      FunctionType newType = FunctionType::get(
          func.getContext(), newInputs, newResults);

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
    module.walk([&](Operation* op) {
      // Skip function ops (handled above)
      if (isa<FuncOp>(op))
        return;

      bool needsUpdate = false;
      for (Type type : op->getResultTypes()) {
        if (isa<TypeVarType>(subst.apply(type)) == false &&
            isa<TypeVarType>(type)) {
          needsUpdate = true;
          break;
        }
        if (subst.apply(type) != type) {
          needsUpdate = true;
          break;
        }
      }

      if (!needsUpdate)
        return;

      // Update result types by creating new operation
      OpBuilder builder(op);
      SmallVector<Type> newResultTypes;
      for (Type type : op->getResultTypes()) {
        newResultTypes.push_back(subst.apply(type));
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
