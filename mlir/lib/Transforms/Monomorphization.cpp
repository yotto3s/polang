//===- Monomorphization.cpp - Function specialization -----------*- C++ -*-===//
//
// This file implements the monomorphization pass for the Polang dialect.
// Monomorphization creates specialized copies of polymorphic functions for
// each unique set of concrete types at call sites.
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
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace polang;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Check if a type is a type variable
bool isTypeVar(Type type) { return isa<TypeVarType>(type); }

/// Check if a function has any type variables in its signature.
/// The entry function (__polang_entry) is never considered polymorphic.
bool isPolymorphicFunction(FuncOp func) {
  // The entry function is never polymorphic - it should always be resolved
  if (func.getSymName() == "__polang_entry")
    return false;

  FunctionType funcType = func.getFunctionType();
  for (Type input : funcType.getInputs()) {
    if (isTypeVar(input))
      return true;
  }
  for (Type result : funcType.getResults()) {
    if (isTypeVar(result))
      return true;
  }
  return false;
}

/// Generate a mangled name for a specialized function
/// Example: identity -> identity$int, identity$int_bool
std::string getMangledName(StringRef baseName, ArrayRef<Type> argTypes) {
  std::string result = baseName.str() + "$";
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (i > 0)
      result += "_";
    if (isa<IntType>(argTypes[i]))
      result += "int";
    else if (isa<DoubleType>(argTypes[i]))
      result += "double";
    else if (isa<BoolType>(argTypes[i]))
      result += "bool";
    else
      result += "unknown";
  }
  return result;
}

/// Generate a signature key for deduplication
std::string getSignatureKey(ArrayRef<Type> argTypes, Type returnType) {
  std::string key;
  for (Type t : argTypes) {
    if (isa<IntType>(t))
      key += "i";
    else if (isa<DoubleType>(t))
      key += "d";
    else if (isa<BoolType>(t))
      key += "b";
    else
      key += "?";
  }
  key += "->";
  if (isa<IntType>(returnType))
    key += "i";
  else if (isa<DoubleType>(returnType))
    key += "d";
  else if (isa<BoolType>(returnType))
    key += "b";
  else
    key += "?";
  return key;
}

/// Build a mapping from type variables to concrete types
void buildTypeVarMapping(FuncOp origFunc, ArrayRef<Type> concreteArgTypes,
                         Type concreteReturnType,
                         llvm::DenseMap<uint64_t, Type>& mapping) {
  FunctionType funcType = origFunc.getFunctionType();

  // Map parameter type variables to concrete types
  for (size_t i = 0; i < funcType.getNumInputs() && i < concreteArgTypes.size();
       ++i) {
    Type paramType = funcType.getInput(i);
    if (auto typeVar = dyn_cast<TypeVarType>(paramType)) {
      mapping[typeVar.getId()] = concreteArgTypes[i];
    }
  }

  // Map return type variable to concrete type
  if (funcType.getNumResults() > 0) {
    Type returnType = funcType.getResult(0);
    if (auto typeVar = dyn_cast<TypeVarType>(returnType)) {
      mapping[typeVar.getId()] = concreteReturnType;
    }
  }
}

/// Apply type variable mapping to a type
Type applyTypeMapping(Type type,
                      const llvm::DenseMap<uint64_t, Type>& mapping) {
  if (auto typeVar = dyn_cast<TypeVarType>(type)) {
    auto it = mapping.find(typeVar.getId());
    if (it != mapping.end())
      return it->second;
  }
  return type;
}

//===----------------------------------------------------------------------===//
// MonomorphizationPass
//===----------------------------------------------------------------------===//

struct MonomorphizationPass
    : public PassWrapper<MonomorphizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MonomorphizationPass)

  StringRef getArgument() const override { return "polang-monomorphize"; }
  StringRef getDescription() const override {
    return "Monomorphize polymorphic functions";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<PolangDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Step 1: Identify all polymorphic functions
    llvm::StringMap<FuncOp> polymorphicFuncs;
    module.walk([&](FuncOp func) {
      if (isPolymorphicFunction(func)) {
        polymorphicFuncs[func.getSymName()] = func;
      }
    });

    if (polymorphicFuncs.empty())
      return; // Nothing to monomorphize

    // Step 2: Collect all unique call signatures for each polymorphic function
    // Map: function name -> (signature key -> (argTypes, returnType))
    struct CallSignature {
      SmallVector<Type> argTypes;
      Type returnType;
    };
    llvm::StringMap<llvm::StringMap<CallSignature>> callSignatures;

    module.walk([&](CallOp call) {
      if (!polymorphicFuncs.contains(call.getCallee()))
        return;

      // Get resolved types from attributes (set by type inference)
      auto resolvedArgTypesAttr =
          call->getAttrOfType<ArrayAttr>("polang.resolved_arg_types");
      auto resolvedReturnTypeAttr =
          call->getAttrOfType<TypeAttr>("polang.resolved_return_type");

      if (!resolvedArgTypesAttr || !resolvedReturnTypeAttr)
        return;

      SmallVector<Type> argTypes;
      for (Attribute attr : resolvedArgTypesAttr) {
        argTypes.push_back(cast<TypeAttr>(attr).getValue());
      }
      Type returnType = resolvedReturnTypeAttr.getValue();

      // Check if this signature already exists
      std::string sigKey = getSignatureKey(argTypes, returnType);
      auto& signatures = callSignatures[call.getCallee()];
      if (!signatures.contains(sigKey)) {
        signatures[sigKey] = {argTypes, returnType};
      }
    });

    // Step 3: Create specialized functions for each unique signature
    // Maps: original name -> (signature key -> specialized name)
    llvm::StringMap<llvm::StringMap<std::string>> specializedNames;

    for (auto& [funcName, signatures] : callSignatures) {
      FuncOp origFunc = polymorphicFuncs[funcName];

      for (auto& [sigKey, sig] : signatures) {
        std::string mangledName = getMangledName(funcName, sig.argTypes);

        // Clone and specialize the function
        FuncOp specializedFunc =
            cloneAndSpecialize(origFunc, mangledName, sig.argTypes,
                               sig.returnType, module);

        if (specializedFunc) {
          specializedNames[funcName][sigKey] = mangledName;
        }
      }
    }

    // Step 4: Update all CallOps to use specialized functions
    module.walk([&](CallOp call) {
      if (!polymorphicFuncs.contains(call.getCallee()))
        return;

      auto resolvedArgTypesAttr =
          call->getAttrOfType<ArrayAttr>("polang.resolved_arg_types");
      auto resolvedReturnTypeAttr =
          call->getAttrOfType<TypeAttr>("polang.resolved_return_type");

      if (!resolvedArgTypesAttr || !resolvedReturnTypeAttr)
        return;

      SmallVector<Type> argTypes;
      for (Attribute attr : resolvedArgTypesAttr) {
        argTypes.push_back(cast<TypeAttr>(attr).getValue());
      }
      Type returnType = resolvedReturnTypeAttr.getValue();

      std::string sigKey = getSignatureKey(argTypes, returnType);

      // Find the specialized function name
      auto funcIt = specializedNames.find(call.getCallee());
      if (funcIt == specializedNames.end())
        return;

      auto sigIt = funcIt->second.find(sigKey);
      if (sigIt == funcIt->second.end())
        return;

      // Update the call to use the specialized function
      call.setCalleeFromCallable(
          SymbolRefAttr::get(call.getContext(), sigIt->second));

      // Update the call's result type if needed
      if (call.getResult() && call.getResult().getType() != returnType) {
        call.getResult().setType(returnType);
      }

      // Remove the temporary attributes
      call->removeAttr("polang.resolved_arg_types");
      call->removeAttr("polang.resolved_return_type");
    });

    // Step 5: Mark original polymorphic functions (for lowering to skip)
    for (auto& [funcName, func] : polymorphicFuncs) {
      func->setAttr("polang.polymorphic", UnitAttr::get(func.getContext()));
    }

    // Step 6: Fix up the __polang_entry function's signature
    // After monomorphization, the return statement may return a value with
    // a concrete type, but the function signature might still have a type var.
    module.walk([&](FuncOp func) {
      if (func.getSymName() != "__polang_entry")
        return;

      FunctionType funcType = func.getFunctionType();
      if (funcType.getNumResults() == 0)
        return;

      Type returnType = funcType.getResult(0);
      if (!isTypeVar(returnType))
        return; // Already concrete

      // Find the return statement and get its actual value type
      Type actualReturnType;
      func.walk([&](ReturnOp returnOp) {
        if (returnOp.getValue()) {
          actualReturnType = returnOp.getValue().getType();
        }
      });

      if (!actualReturnType || isTypeVar(actualReturnType))
        return; // Couldn't determine concrete type

      // Update the function signature
      FunctionType newFuncType = FunctionType::get(
          func.getContext(), funcType.getInputs(), {actualReturnType});
      func.setType(newFuncType);
    });
  }

private:
  /// Clone a polymorphic function and specialize it with concrete types
  FuncOp cloneAndSpecialize(FuncOp origFunc, StringRef newName,
                            ArrayRef<Type> concreteArgTypes,
                            Type concreteReturnType, ModuleOp module) {
    OpBuilder builder(module.getContext());

    // Build mapping from type variables to concrete types
    llvm::DenseMap<uint64_t, Type> typeMapping;
    buildTypeVarMapping(origFunc, concreteArgTypes, concreteReturnType,
                        typeMapping);

    // Create new function type with concrete types
    FunctionType newFuncType =
        builder.getFunctionType(concreteArgTypes, {concreteReturnType});

    // Create the new function
    builder.setInsertionPointAfter(origFunc);
    auto newFunc =
        builder.create<FuncOp>(origFunc.getLoc(), newName, newFuncType);

    // Clone the body
    IRMapping mapping;
    origFunc.getBody().cloneInto(&newFunc.getBody(), mapping);

    // Update block argument types
    if (!newFunc.getBody().empty()) {
      Block& entry = newFunc.getBody().front();
      for (size_t i = 0; i < entry.getNumArguments() &&
                         i < concreteArgTypes.size();
           ++i) {
        entry.getArgument(i).setType(concreteArgTypes[i]);
      }
    }

    // Update types of all operations in the body
    updateFunctionBodyTypes(newFunc, typeMapping, origFunc.getSymName(),
                            newName);

    return newFunc;
  }

  /// Update types of operations in the specialized function body
  void updateFunctionBodyTypes(FuncOp func,
                               const llvm::DenseMap<uint64_t, Type>& typeMapping,
                               StringRef origFuncName, StringRef newFuncName) {
    // Collect operations that need type updates (we can't modify while walking)
    SmallVector<Operation*> opsToUpdate;
    func.walk([&](Operation* op) {
      if (isa<FuncOp>(op))
        return;

      bool needsUpdate = false;
      for (Type type : op->getResultTypes()) {
        if (isTypeVar(type)) {
          needsUpdate = true;
          break;
        }
      }
      if (needsUpdate)
        opsToUpdate.push_back(op);
    });

    // Update operations
    for (Operation* op : opsToUpdate) {
      // Handle CallOp specially - update recursive calls
      if (auto callOp = dyn_cast<CallOp>(op)) {
        // If this is a recursive call to the original function, update it
        if (callOp.getCallee() == origFuncName) {
          callOp.setCalleeFromCallable(
              SymbolRefAttr::get(func.getContext(), newFuncName));
        }

        // Update result type if it's a type variable
        if (callOp.getResult()) {
          Type resultType = callOp.getResult().getType();
          Type newType = applyTypeMapping(resultType, typeMapping);
          if (newType != resultType) {
            callOp.getResult().setType(newType);
          }
        }
        continue;
      }

      // For other operations, rebuild with updated types
      OpBuilder builder(op);
      SmallVector<Type> newResultTypes;
      for (Type type : op->getResultTypes()) {
        newResultTypes.push_back(applyTypeMapping(type, typeMapping));
      }

      // Create operation state
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

      // Replace uses
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
      }

      op->erase();
    }
  }
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createMonomorphizationPass() {
  return std::make_unique<MonomorphizationPass>();
}

} // namespace polang
