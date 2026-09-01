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

#include "llvm/ADT/StringMap.h"

#pragma GCC diagnostic pop

using namespace mlir;
using namespace polang;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Get a type string for mangling
std::string getTypeString(Type type) {
  if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
    if (intType.getWidth() == 1) {
      return "bool";
    }
    return (intType.isUnsigned() ? "u" : "i") +
           std::to_string(intType.getWidth());
  }
  if (auto floatType = dyn_cast<mlir::FloatType>(type)) {
    return "f" + std::to_string(floatType.getWidth());
  }
  if (isa<mlir::IndexType>(type)) {
    return "index";
  }
  if (auto tupleType = dyn_cast<polang::TupleType>(type)) {
    size_t n = tupleType.getTypes().size();
    if (n == 0) {
      return "unit";
    }
    std::string s = "tup" + std::to_string(n);
    for (Type elem : tupleType.getTypes()) {
      s += "_" + getTypeString(elem);
    }
    return s;
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

/// Check if a type is a TypeParamType
bool isTypeParam(Type type) {
  if (isa<TypeParamType>(type)) {
    return true;
  }
  if (auto tupleType = dyn_cast<polang::TupleType>(type)) {
    for (Type elem : tupleType.getTypes()) {
      if (isTypeParam(elem)) {
        return true;
      }
    }
  }
  return false;
}

/// Apply name-based type parameter mapping to a type
Type applyTypeParamMapping(Type type, const llvm::StringMap<Type>& mapping) {
  if (auto paramType = dyn_cast<TypeParamType>(type)) {
    auto it = mapping.find(paramType.getName());
    if (it != mapping.end()) {
      return it->second;
    }
  }
  if (auto tupleType = dyn_cast<polang::TupleType>(type)) {
    SmallVector<Type> mappedTypes;
    bool changed = false;
    for (Type elem : tupleType.getTypes()) {
      Type mappedElem = applyTypeParamMapping(elem, mapping);
      if (mappedElem != elem) {
        changed = true;
      }
      mappedTypes.push_back(mappedElem);
    }
    if (changed) {
      return polang::TupleType::get(type.getContext(), mappedTypes);
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

    // Handle GenericFuncOp + InstantiateOp (name-based type params)
    monomorphizeGenericFunctions(module);
  }

private:
  /// Signature info for a GenericFuncOp instantiation.
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
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createMonomorphizationPass() {
  return std::make_unique<MonomorphizationPass>();
}

void registerPolangTransformsPasses() {
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return polang::createMonomorphizationPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return polang::createCheckConstantOverflowPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return polang::createInsertOverflowChecksPass();
  });
}

} // namespace polang
