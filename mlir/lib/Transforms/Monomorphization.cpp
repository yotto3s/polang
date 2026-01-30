//===- Monomorphization.cpp - Function specialization -----------*- C++ -*-===//
//
// This file implements the monomorphization pass for the Polang dialect.
// Monomorphization creates specialized copies of polymorphic functions for
// each unique set of concrete types at call sites.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"
#include "polang/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#pragma GCC diagnostic pop

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
  if (func.getSymName() == "__polang_entry") {
    return false;
  }

  FunctionType funcType = func.getFunctionType();
  if (llvm::any_of(funcType.getInputs(),
                   [](Type input) { return isTypeVar(input); })) {
    return true;
  }
  return llvm::any_of(funcType.getResults(),
                      [](Type result) { return isTypeVar(result); });
}

/// Get a type string for mangling
std::string getTypeString(Type type) {
  if (auto intType = dyn_cast<polang::IntegerType>(type)) {
    return (intType.isSigned() ? "i" : "u") +
           std::to_string(intType.getWidth());
  }
  if (auto floatType = dyn_cast<polang::FloatType>(type)) {
    return "f" + std::to_string(floatType.getWidth());
  }
  if (isa<BoolType>(type)) {
    return "bool";
  }
  return "unknown";
}

/// Generate a mangled name for a specialized function
/// Example: identity -> identity$i64, identity$i64_bool
std::string getMangledName(StringRef baseName, ArrayRef<Type> argTypes) {
  std::string result = baseName.str() + "$";
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (i > 0) {
      result += "_";
    }
    result += getTypeString(argTypes[i]);
  }
  return result;
}

/// Generate a signature key for deduplication
std::string getSignatureKey(ArrayRef<Type> argTypes, Type returnType) {
  std::string key;
  for (Type t : argTypes) {
    key += getTypeString(t) + ",";
  }
  key += "->";
  key += getTypeString(returnType);
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
    if (it != mapping.end()) {
      return it->second;
    }
  }
  return type;
}

/// Check if a type is a TypeParamType
bool isTypeParam(Type type) { return isa<TypeParamType>(type); }

/// Apply name-based type parameter mapping to a type
Type applyTypeParamMapping(Type type, const llvm::StringMap<Type>& mapping) {
  if (auto paramType = dyn_cast<TypeParamType>(type)) {
    auto it = mapping.find(paramType.getName());
    if (it != mapping.end()) {
      return it->second;
    }
  }
  return type;
}

//===----------------------------------------------------------------------===//
// MonomorphizationPass
//===----------------------------------------------------------------------===//

struct MonomorphizationPass
    : public PassWrapper<MonomorphizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MonomorphizationPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "polang-monomorphize";
  }
  [[nodiscard]] StringRef getDescription() const override {
    return "Monomorphize polymorphic functions";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<PolangDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // NEW: Handle GenericFuncOp + InstantiateOp (name-based type params)
    monomorphizeGenericFunctions(module);

    // EXISTING: Handle FuncOp + TypeVarType (ID-based, legacy)
    auto polymorphicFuncs = identifyPolymorphicFunctions(module);
    if (polymorphicFuncs.empty()) {
      return;
    }

    auto callSignatures = collectCallSignatures(module, polymorphicFuncs);
    auto specializedNames =
        createSpecializedFunctions(module, polymorphicFuncs, callSignatures);
    updateCallsToSpecialized(module, polymorphicFuncs, specializedNames);
    markPolymorphicFunctions(polymorphicFuncs);
    fixupEntryFunctionSignature(module);
  }

private:
  /// Signature info for a monomorphized call site (legacy TypeVarType path).
  struct CallSignature {
    SmallVector<Type> argTypes;
    Type returnType;
  };

  /// Signature info for a GenericFuncOp instantiation (name-based path).
  struct GenericBindingSignature {
    SmallVector<Type> concreteTypes; // concrete types for each type param
    SmallVector<Type> argTypes;      // fully resolved arg types
    Type returnType;                 // fully resolved return type
  };

  //===--------------------------------------------------------------------===//
  // GenericFuncOp + InstantiateOp monomorphization (name-based type params)
  //===--------------------------------------------------------------------===//

  /// Orchestrate the GenericFuncOp monomorphization path.
  void monomorphizeGenericFunctions(ModuleOp module) {
    // 1. Collect all GenericFuncOp instances
    llvm::StringMap<GenericFuncOp> genericFuncs;
    module.walk([&](GenericFuncOp op) { genericFuncs[op.getSymName()] = op; });

    if (genericFuncs.empty()) {
      return;
    }

    // 2. Collect all unique bindings from InstantiateOp instances
    auto bindings = collectGenericBindings(module, genericFuncs);

    // 3. For each GenericFuncOp with call sites, create specialized functions
    // Maps original function name + signature key -> mangled name
    llvm::StringMap<llvm::StringMap<std::string>> mangledNames;

    for (const auto& [funcName, signatures] : bindings) {
      auto funcIt = genericFuncs.find(funcName);
      if (funcIt == genericFuncs.end()) {
        continue;
      }
      GenericFuncOp genericFunc = funcIt->second;

      for (const auto& [sigKey, sig] : signatures) {
        std::string mangledName = getMangledName(funcName, sig.argTypes);

        // Build type param mapping
        llvm::StringMap<Type> typeParamMapping;
        auto typeParams = genericFunc.getTypeParams();
        for (size_t i = 0;
             i < typeParams.size() && i < sig.concreteTypes.size(); ++i) {
          StringRef paramName = cast<StringAttr>(typeParams[i]).getValue();
          typeParamMapping[paramName] = sig.concreteTypes[i];
        }

        FuncOp specialized = cloneAndSpecializeGeneric(
            genericFunc, mangledName, typeParamMapping, sig.argTypes,
            sig.returnType, module);

        if (specialized) {
          mangledNames[funcName][sigKey] = mangledName;
        }
      }
    }

    // 4. Replace InstantiateOps with CallOps
    replaceInstantiateOps(module, genericFuncs, mangledNames);

    // 5. Erase original GenericFuncOp instances
    for (auto& [funcName, genericFunc] : genericFuncs) {
      genericFunc.erase();
    }
  }

  /// Collect all unique instantiation bindings from InstantiateOp instances.
  [[nodiscard]] llvm::StringMap<llvm::StringMap<GenericBindingSignature>>
  collectGenericBindings(ModuleOp module,
                         const llvm::StringMap<GenericFuncOp>& genericFuncs) {
    llvm::StringMap<llvm::StringMap<GenericBindingSignature>> result;

    module.walk([&](InstantiateOp instantiate) {
      StringRef callee = instantiate.getCallee();
      if (!genericFuncs.contains(callee)) {
        return;
      }

      auto typeParamNames = instantiate.getTypeParamNames();
      auto typeBindingsAttr = instantiate.getTypeBindings();

      SmallVector<Type> concreteTypes;
      for (Attribute attr : typeBindingsAttr) {
        concreteTypes.push_back(cast<TypeAttr>(attr).getValue());
      }

      // Build concrete arg types by substituting type params in the
      // GenericFuncOp signature
      GenericFuncOp genericFunc = genericFuncs.lookup(callee);
      FunctionType funcType = genericFunc.getFunctionType();

      // Build a name->type mapping from the instantiation
      llvm::StringMap<Type> paramMapping;
      for (size_t i = 0; i < typeParamNames.size(); ++i) {
        StringRef name = cast<StringAttr>(typeParamNames[i]).getValue();
        paramMapping[name] = concreteTypes[i];
      }

      SmallVector<Type> argTypes;
      for (Type inputType : funcType.getInputs()) {
        argTypes.push_back(applyTypeParamMapping(inputType, paramMapping));
      }

      Type returnType =
          funcType.getNumResults() > 0
              ? applyTypeParamMapping(funcType.getResult(0), paramMapping)
              : Type();

      std::string sigKey = getSignatureKey(argTypes, returnType);
      auto& signatures = result[callee];
      if (!signatures.contains(sigKey)) {
        signatures[sigKey] = {concreteTypes, argTypes, returnType};
      }
    });

    return result;
  }

  /// Clone a GenericFuncOp and specialize it with concrete types.
  FuncOp
  cloneAndSpecializeGeneric(GenericFuncOp genericFunc, StringRef newName,
                            const llvm::StringMap<Type>& typeParamMapping,
                            ArrayRef<Type> concreteArgTypes,
                            Type concreteReturnType, ModuleOp module) {
    OpBuilder builder(module.getContext());

    // Create new function type with concrete types
    SmallVector<Type> resultTypes;
    if (concreteReturnType) {
      resultTypes.push_back(concreteReturnType);
    }
    FunctionType newFuncType =
        builder.getFunctionType(concreteArgTypes, resultTypes);

    // Create the new FuncOp
    builder.setInsertionPointAfter(genericFunc);
    auto newFunc =
        builder.create<FuncOp>(genericFunc.getLoc(), newName, newFuncType);

    // Clone the body
    IRMapping mapping;
    genericFunc.getBody().cloneInto(&newFunc.getBody(), mapping);

    // Update block argument types
    if (!newFunc.getBody().empty()) {
      Block& entry = newFunc.getBody().front();
      for (size_t i = 0;
           i < entry.getNumArguments() && i < concreteArgTypes.size(); ++i) {
        entry.getArgument(i).setType(concreteArgTypes[i]);
      }
    }

    // Update types of all operations in the body
    updateGenericFunctionBodyTypes(newFunc, typeParamMapping,
                                   genericFunc.getSymName(), newName);

    return newFunc;
  }

  /// Update types of operations in a specialized function body.
  /// Uses name-based TypeParamType mapping.
  void updateGenericFunctionBodyTypes(
      FuncOp func, const llvm::StringMap<Type>& typeParamMapping,
      StringRef origFuncName, StringRef newFuncName) {
    // Collect operations that need type updates
    SmallVector<Operation*> opsToUpdate;
    func.walk([&](Operation* op) {
      if (isa<FuncOp>(op)) {
        return;
      }

      bool needsUpdate = false;
      for (Type type : op->getResultTypes()) {
        if (isTypeParam(type)) {
          needsUpdate = true;
          break;
        }
      }
      if (needsUpdate) {
        opsToUpdate.push_back(op);
      }
    });

    // Update operations
    for (Operation* op : opsToUpdate) {
      // Handle InstantiateOp - recursive calls to the same generic function
      if (auto instantiateOp = dyn_cast<InstantiateOp>(op)) {
        if (instantiateOp.getCallee() == origFuncName) {
          // Replace with a CallOp to the specialized function
          OpBuilder builder(instantiateOp);
          SmallVector<Type> resultTypes;
          if (instantiateOp.getNumResults() > 0) {
            Type resultType = applyTypeParamMapping(
                instantiateOp.getResult().getType(), typeParamMapping);
            resultTypes.push_back(resultType);
          }
          auto callOp =
              builder.create<CallOp>(instantiateOp.getLoc(), newFuncName,
                                     resultTypes, instantiateOp.getOperands());
          if (instantiateOp.getNumResults() > 0) {
            instantiateOp.getResult().replaceAllUsesWith(callOp.getResult());
          }
          instantiateOp.erase();
          continue;
        }
      }

      // Handle CallOp - update recursive calls
      if (auto callOp = dyn_cast<CallOp>(op)) {
        if (callOp.getCallee() == origFuncName) {
          callOp.setCalleeFromCallable(
              SymbolRefAttr::get(func.getContext(), newFuncName));
        }

        // Update result type if it's a type parameter
        if (callOp.getResult()) {
          Type resultType = callOp.getResult().getType();
          Type newType = applyTypeParamMapping(resultType, typeParamMapping);
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
        newResultTypes.push_back(applyTypeParamMapping(type, typeParamMapping));
      }

      OperationState state(op->getLoc(), op->getName());
      state.addOperands(op->getOperands());
      state.addTypes(newResultTypes);
      state.addAttributes(op->getAttrs());

      for (Region& region : op->getRegions()) {
        Region* newRegion = state.addRegion();
        IRMapping regionMapping;
        region.cloneInto(newRegion, regionMapping);
      }

      Operation* newOp = builder.create(state);

      for (size_t i = 0; i < op->getNumResults(); ++i) {
        op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
      }

      op->erase();
    }
  }

  /// Replace all InstantiateOps with CallOps to specialized functions.
  void replaceInstantiateOps(
      ModuleOp module, const llvm::StringMap<GenericFuncOp>& genericFuncs,
      const llvm::StringMap<llvm::StringMap<std::string>>& mangledNames) {
    // Collect InstantiateOps first to avoid invalidating the walk iterator
    SmallVector<InstantiateOp> instantiateOps;
    module.walk([&](InstantiateOp op) { instantiateOps.push_back(op); });

    for (InstantiateOp instantiate : instantiateOps) {
      StringRef callee = instantiate.getCallee();
      auto funcIt = mangledNames.find(callee);
      if (funcIt == mangledNames.end()) {
        continue;
      }

      // Reconstruct the signature key to find the mangled name
      auto genericFuncIt = genericFuncs.find(callee);
      if (genericFuncIt == genericFuncs.end()) {
        continue;
      }
      GenericFuncOp genericFunc = genericFuncIt->second;
      FunctionType funcType = genericFunc.getFunctionType();

      // Build param mapping from the instantiate op
      auto typeParamNames = instantiate.getTypeParamNames();
      auto typeBindingsAttr = instantiate.getTypeBindings();
      llvm::StringMap<Type> paramMapping;
      for (size_t i = 0; i < typeParamNames.size(); ++i) {
        StringRef name = cast<StringAttr>(typeParamNames[i]).getValue();
        paramMapping[name] = cast<TypeAttr>(typeBindingsAttr[i]).getValue();
      }

      SmallVector<Type> argTypes;
      for (Type inputType : funcType.getInputs()) {
        argTypes.push_back(applyTypeParamMapping(inputType, paramMapping));
      }
      Type returnType =
          funcType.getNumResults() > 0
              ? applyTypeParamMapping(funcType.getResult(0), paramMapping)
              : Type();

      std::string sigKey = getSignatureKey(argTypes, returnType);
      auto sigIt = funcIt->second.find(sigKey);
      if (sigIt == funcIt->second.end()) {
        continue;
      }

      // Create CallOp to the specialized function
      OpBuilder builder(instantiate);
      SmallVector<Type> resultTypes;
      if (returnType) {
        resultTypes.push_back(returnType);
      }
      auto callOp =
          builder.create<CallOp>(instantiate.getLoc(), sigIt->second,
                                 resultTypes, instantiate.getOperands());

      if (instantiate.getNumResults() > 0 && callOp.getNumResults() > 0) {
        instantiate.getResult().replaceAllUsesWith(callOp.getResult());
      }

      instantiate.erase();
    }
  }

  //===--------------------------------------------------------------------===//
  // Legacy FuncOp + TypeVarType monomorphization (ID-based)
  //===--------------------------------------------------------------------===//

  /// Identify all polymorphic functions in the module.
  [[nodiscard]] llvm::StringMap<FuncOp>
  identifyPolymorphicFunctions(ModuleOp module) {
    llvm::StringMap<FuncOp> polymorphicFuncs;
    module.walk([&](FuncOp func) {
      if (isPolymorphicFunction(func)) {
        polymorphicFuncs[func.getSymName()] = func;
      }
    });
    return polymorphicFuncs;
  }

  /// Collect all unique call signatures for each polymorphic function.
  [[nodiscard]] llvm::StringMap<llvm::StringMap<CallSignature>>
  collectCallSignatures(ModuleOp module,
                        const llvm::StringMap<FuncOp>& polymorphicFuncs) {
    llvm::StringMap<llvm::StringMap<CallSignature>> callSignatures;

    module.walk([&](CallOp call) {
      if (!polymorphicFuncs.contains(call.getCallee())) {
        return;
      }

      auto resolvedArgTypesAttr =
          call->getAttrOfType<ArrayAttr>("polang.resolved_arg_types");
      auto resolvedReturnTypeAttr =
          call->getAttrOfType<TypeAttr>("polang.resolved_return_type");

      if (!resolvedArgTypesAttr || !resolvedReturnTypeAttr) {
        return;
      }

      SmallVector<Type> argTypes;
      for (Attribute attr : resolvedArgTypesAttr) {
        argTypes.push_back(cast<TypeAttr>(attr).getValue());
      }
      Type returnType = resolvedReturnTypeAttr.getValue();

      std::string sigKey = getSignatureKey(argTypes, returnType);
      auto& signatures = callSignatures[call.getCallee()];
      if (!signatures.contains(sigKey)) {
        signatures[sigKey] = {argTypes, returnType};
      }
    });

    return callSignatures;
  }

  /// Create specialized functions for each unique signature.
  [[nodiscard]] llvm::StringMap<llvm::StringMap<std::string>>
  createSpecializedFunctions(
      ModuleOp module, const llvm::StringMap<FuncOp>& polymorphicFuncs,
      const llvm::StringMap<llvm::StringMap<CallSignature>>& callSignatures) {
    llvm::StringMap<llvm::StringMap<std::string>> specializedNames;

    for (const auto& [funcName, signatures] : callSignatures) {
      auto funcIt = polymorphicFuncs.find(funcName);
      if (funcIt == polymorphicFuncs.end()) {
        continue;
      }
      FuncOp origFunc = funcIt->second;

      for (const auto& [sigKey, sig] : signatures) {
        std::string mangledName = getMangledName(funcName, sig.argTypes);

        FuncOp specializedFunc = cloneAndSpecialize(
            origFunc, mangledName, sig.argTypes, sig.returnType, module);

        if (specializedFunc) {
          specializedNames[funcName][sigKey] = mangledName;
        }
      }
    }

    return specializedNames;
  }

  /// Update all CallOps to use specialized functions.
  void updateCallsToSpecialized(
      ModuleOp module, const llvm::StringMap<FuncOp>& polymorphicFuncs,
      const llvm::StringMap<llvm::StringMap<std::string>>& specializedNames) {
    module.walk([&](CallOp call) {
      if (!polymorphicFuncs.contains(call.getCallee())) {
        return;
      }

      auto resolvedArgTypesAttr =
          call->getAttrOfType<ArrayAttr>("polang.resolved_arg_types");
      auto resolvedReturnTypeAttr =
          call->getAttrOfType<TypeAttr>("polang.resolved_return_type");

      if (!resolvedArgTypesAttr || !resolvedReturnTypeAttr) {
        return;
      }

      SmallVector<Type> argTypes;
      for (Attribute attr : resolvedArgTypesAttr) {
        argTypes.push_back(cast<TypeAttr>(attr).getValue());
      }
      Type returnType = resolvedReturnTypeAttr.getValue();

      std::string sigKey = getSignatureKey(argTypes, returnType);

      auto funcIt = specializedNames.find(call.getCallee());
      if (funcIt == specializedNames.end()) {
        return;
      }

      auto sigIt = funcIt->second.find(sigKey);
      if (sigIt == funcIt->second.end()) {
        return;
      }

      call.setCalleeFromCallable(
          SymbolRefAttr::get(call.getContext(), sigIt->second));

      if (call.getResult() && call.getResult().getType() != returnType) {
        call.getResult().setType(returnType);
      }

      call->removeAttr("polang.resolved_arg_types");
      call->removeAttr("polang.resolved_return_type");
    });
  }

  /// Mark original polymorphic functions for lowering to skip.
  void markPolymorphicFunctions(llvm::StringMap<FuncOp>& polymorphicFuncs) {
    for (auto& [funcName, func] : polymorphicFuncs) {
      func->setAttr("polang.polymorphic", UnitAttr::get(func.getContext()));
    }
  }

  /// Fix up the __polang_entry function's signature if it has type variables.
  void fixupEntryFunctionSignature(ModuleOp module) {
    module.walk([&](FuncOp func) {
      if (func.getSymName() != "__polang_entry") {
        return;
      }

      FunctionType funcType = func.getFunctionType();
      if (funcType.getNumResults() == 0) {
        return;
      }

      Type returnType = funcType.getResult(0);
      if (!isTypeVar(returnType)) {
        return;
      }

      Type actualReturnType;
      func.walk([&](ReturnOp returnOp) {
        if (returnOp.getValue()) {
          actualReturnType = returnOp.getValue().getType();
        }
      });

      if (!actualReturnType || isTypeVar(actualReturnType)) {
        return;
      }

      FunctionType newFuncType = FunctionType::get(
          func.getContext(), funcType.getInputs(), {actualReturnType});
      func.setType(newFuncType);
    });
  }

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
      for (size_t i = 0;
           i < entry.getNumArguments() && i < concreteArgTypes.size(); ++i) {
        entry.getArgument(i).setType(concreteArgTypes[i]);
      }
    }

    // Update types of all operations in the body
    updateFunctionBodyTypes(newFunc, typeMapping, origFunc.getSymName(),
                            newName);

    return newFunc;
  }

  /// Update types of operations in the specialized function body
  void
  updateFunctionBodyTypes(FuncOp func,
                          const llvm::DenseMap<uint64_t, Type>& typeMapping,
                          StringRef origFuncName, StringRef newFuncName) {
    // Collect operations that need type updates (we can't modify while walking)
    SmallVector<Operation*> opsToUpdate;
    func.walk([&](Operation* op) {
      if (isa<FuncOp>(op)) {
        return;
      }

      bool needsUpdate = false;
      for (Type type : op->getResultTypes()) {
        if (isTypeVar(type)) {
          needsUpdate = true;
          break;
        }
      }
      if (needsUpdate) {
        opsToUpdate.push_back(op);
      }
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
