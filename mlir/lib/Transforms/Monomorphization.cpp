//===- Monomorphization.cpp - Function specialization -----------*- C++ -*-===//
//
// This file implements the monomorphization pass for the Polang dialect.
// The pass supports two formats:
//
// 1. New format: GenericFuncOp with TypeParamType and CallOp with type_args
//    - Creates SpecializedFuncOp markers (body cloned during lowering)
//
// 2. Legacy format: FuncOp with TypeVarType and polang.resolved_* attributes
//    - Creates specialized FuncOp copies with concrete types
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"
#include "polang/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#pragma GCC diagnostic pop

using namespace mlir;
using namespace polang;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Check if a type is a type variable
bool isTypeVar(Type type) { return isa<TypeVarType>(type); }

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
std::string getMangledName(StringRef baseName, ArrayRef<Type> typeArgs) {
  std::string result = baseName.str() + "$";
  for (size_t i = 0; i < typeArgs.size(); ++i) {
    if (i > 0) {
      result += "_";
    }
    result += getTypeString(typeArgs[i]);
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

/// Generate a type args key for deduplication (based on type args)
std::string getTypeArgsKey(ArrayRef<Type> typeArgs) {
  std::string key;
  for (size_t i = 0; i < typeArgs.size(); ++i) {
    if (i > 0) {
      key += ",";
    }
    key += getTypeString(typeArgs[i]);
  }
  return key;
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

    // Try new format first (GenericFuncOp + type_args)
    if (processNewFormat(module)) {
      return;
    }

    // Fall back to legacy format (FuncOp + TypeVarType + resolved_* attrs)
    processLegacyFormat(module);
  }

private:
  //=========================================================================//
  // New Format: GenericFuncOp + type_args
  //=========================================================================//

  /// Instantiation info for a monomorphized call site.
  struct Instantiation {
    SmallVector<Type> typeArgs;
  };

  /// Process the new format (GenericFuncOp with type_args).
  /// Returns true if any GenericFuncOp was found and processed.
  bool processNewFormat(ModuleOp module) {
    auto genericFuncs = collectGenericFunctions(module);
    if (genericFuncs.empty()) {
      return false;
    }

    auto instantiations = collectInstantiations(module, genericFuncs);
    auto specializedNames =
        createSpecializedFuncs(module, genericFuncs, instantiations);
    updateCallSitesNewFormat(module, genericFuncs, specializedNames);
    return true;
  }

  /// Collect all GenericFuncOp in the module.
  [[nodiscard]] llvm::StringMap<GenericFuncOp>
  collectGenericFunctions(ModuleOp module) {
    llvm::StringMap<GenericFuncOp> genericFuncs;
    module.walk([&](GenericFuncOp func) {
      genericFuncs[func.getSymName()] = func;
    });
    return genericFuncs;
  }

  /// Collect all unique type arg instantiations for each generic function.
  [[nodiscard]] llvm::StringMap<llvm::StringMap<Instantiation>>
  collectInstantiations(ModuleOp module,
                        const llvm::StringMap<GenericFuncOp>& genericFuncs) {
    llvm::StringMap<llvm::StringMap<Instantiation>> instantiations;

    module.walk([&](CallOp call) {
      if (!call.getTypeArgs()) {
        return;
      }

      StringRef callee = call.getCallee();
      if (!genericFuncs.contains(callee)) {
        return;
      }

      SmallVector<Type> typeArgs;
      ArrayAttr typeArgsAttr = *call.getTypeArgs();
      for (Attribute attr : typeArgsAttr) {
        typeArgs.push_back(cast<TypeAttr>(attr).getValue());
      }

      std::string key = getTypeArgsKey(typeArgs);
      auto& funcInstantiations = instantiations[callee];
      if (!funcInstantiations.contains(key)) {
        funcInstantiations[key] = {typeArgs};
      }
    });

    return instantiations;
  }

  /// Create SpecializedFuncOp markers for each unique instantiation.
  [[nodiscard]] llvm::StringMap<llvm::StringMap<std::string>>
  createSpecializedFuncs(
      ModuleOp module, const llvm::StringMap<GenericFuncOp>& genericFuncs,
      const llvm::StringMap<llvm::StringMap<Instantiation>>& instantiations) {
    llvm::StringMap<llvm::StringMap<std::string>> specializedNames;
    OpBuilder builder(module.getContext());

    for (const auto& [funcName, funcInstantiations] : instantiations) {
      auto funcIt = genericFuncs.find(funcName);
      if (funcIt == genericFuncs.end()) {
        continue;
      }
      GenericFuncOp genericFunc = funcIt->second;

      builder.setInsertionPointAfter(genericFunc);

      for (const auto& [key, inst] : funcInstantiations) {
        std::string mangledName = getMangledName(funcName, inst.typeArgs);

        SmallVector<Attribute> typeArgAttrs;
        for (Type t : inst.typeArgs) {
          typeArgAttrs.push_back(TypeAttr::get(t));
        }
        ArrayAttr typeArgsAttr = builder.getArrayAttr(typeArgAttrs);

        builder.create<SpecializedFuncOp>(genericFunc.getLoc(), mangledName,
                                          funcName, typeArgsAttr);

        specializedNames[funcName][key] = mangledName;
      }
    }

    return specializedNames;
  }

  /// Update all CallOps to use specialized function names (new format).
  void updateCallSitesNewFormat(
      ModuleOp module, const llvm::StringMap<GenericFuncOp>& genericFuncs,
      const llvm::StringMap<llvm::StringMap<std::string>>& specializedNames) {
    module.walk([&](CallOp call) {
      if (!call.getTypeArgs()) {
        return;
      }

      StringRef callee = call.getCallee();
      if (!genericFuncs.contains(callee)) {
        return;
      }

      auto funcIt = specializedNames.find(callee);
      if (funcIt == specializedNames.end()) {
        return;
      }

      SmallVector<Type> typeArgs;
      ArrayAttr callTypeArgs = *call.getTypeArgs();
      for (Attribute attr : callTypeArgs) {
        typeArgs.push_back(cast<TypeAttr>(attr).getValue());
      }
      std::string key = getTypeArgsKey(typeArgs);

      auto nameIt = funcIt->second.find(key);
      if (nameIt == funcIt->second.end()) {
        return;
      }

      call.setCalleeFromCallable(
          SymbolRefAttr::get(call.getContext(), nameIt->second));
      call->removeAttr("type_args");
    });
  }

  //=========================================================================//
  // Legacy Format: FuncOp + TypeVarType + resolved_* attrs
  //=========================================================================//

  /// Signature info for a monomorphized call site.
  struct CallSignature {
    SmallVector<Type> argTypes;
    Type returnType;
  };

  /// Check if a function is polymorphic (has type variables in signature).
  [[nodiscard]] bool isPolymorphicFunction(FuncOp func) {
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

  /// Process the legacy format (FuncOp + TypeVarType).
  void processLegacyFormat(ModuleOp module) {
    auto polymorphicFuncs = identifyPolymorphicFunctions(module);
    if (polymorphicFuncs.empty()) {
      return;
    }

    auto callSignatures = collectCallSignaturesLegacy(module, polymorphicFuncs);
    auto specializedNames = createSpecializedFunctionsLegacy(
        module, polymorphicFuncs, callSignatures);
    updateCallSitesLegacy(module, polymorphicFuncs, specializedNames);
    markPolymorphicFunctions(polymorphicFuncs);
    fixupEntryFunctionSignature(module);
  }

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
  collectCallSignaturesLegacy(
      ModuleOp module, const llvm::StringMap<FuncOp>& polymorphicFuncs) {
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
  createSpecializedFunctionsLegacy(
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

  /// Update all CallOps to use specialized functions (legacy format).
  void updateCallSitesLegacy(
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

  /// Build a mapping from type variables to concrete types
  void buildTypeVarMapping(FuncOp origFunc, ArrayRef<Type> concreteArgTypes,
                           Type concreteReturnType,
                           llvm::DenseMap<uint64_t, Type>& mapping) {
    FunctionType funcType = origFunc.getFunctionType();

    for (size_t i = 0;
         i < funcType.getNumInputs() && i < concreteArgTypes.size(); ++i) {
      Type paramType = funcType.getInput(i);
      if (auto typeVar = dyn_cast<TypeVarType>(paramType)) {
        mapping[typeVar.getId()] = concreteArgTypes[i];
      }
    }

    if (funcType.getNumResults() > 0) {
      Type returnType = funcType.getResult(0);
      if (auto typeVar = dyn_cast<TypeVarType>(returnType)) {
        mapping[typeVar.getId()] = concreteReturnType;
      }
    }
  }

  /// Clone a polymorphic function and specialize it with concrete types
  FuncOp cloneAndSpecialize(FuncOp origFunc, StringRef newName,
                            ArrayRef<Type> concreteArgTypes,
                            Type concreteReturnType, ModuleOp module) {
    OpBuilder builder(module.getContext());

    llvm::DenseMap<uint64_t, Type> typeMapping;
    buildTypeVarMapping(origFunc, concreteArgTypes, concreteReturnType,
                        typeMapping);

    FunctionType newFuncType =
        builder.getFunctionType(concreteArgTypes, {concreteReturnType});

    builder.setInsertionPointAfter(origFunc);
    auto newFunc =
        builder.create<FuncOp>(origFunc.getLoc(), newName, newFuncType);

    IRMapping mapping;
    origFunc.getBody().cloneInto(&newFunc.getBody(), mapping);

    if (!newFunc.getBody().empty()) {
      Block& entry = newFunc.getBody().front();
      for (size_t i = 0;
           i < entry.getNumArguments() && i < concreteArgTypes.size(); ++i) {
        entry.getArgument(i).setType(concreteArgTypes[i]);
      }
    }

    updateFunctionBodyTypes(newFunc, typeMapping, origFunc.getSymName(),
                            newName);

    return newFunc;
  }

  /// Update types of operations in the specialized function body
  void
  updateFunctionBodyTypes(FuncOp func,
                          const llvm::DenseMap<uint64_t, Type>& typeMapping,
                          StringRef origFuncName, StringRef newFuncName) {
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

    for (Operation* op : opsToUpdate) {
      if (auto callOp = dyn_cast<CallOp>(op)) {
        if (callOp.getCallee() == origFuncName) {
          callOp.setCalleeFromCallable(
              SymbolRefAttr::get(func.getContext(), newFuncName));
        }

        if (callOp.getResult()) {
          Type resultType = callOp.getResult().getType();
          Type newType = applyTypeMapping(resultType, typeMapping);
          if (newType != resultType) {
            callOp.getResult().setType(newType);
          }
        }
        continue;
      }

      OpBuilder builder(op);
      SmallVector<Type> newResultTypes;
      for (Type type : op->getResultTypes()) {
        newResultTypes.push_back(applyTypeMapping(type, typeMapping));
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
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createMonomorphizationPass() {
  return std::make_unique<MonomorphizationPass>();
}

} // namespace polang
