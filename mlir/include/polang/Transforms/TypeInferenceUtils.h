//===- TypeInferenceUtils.h - Type inference utilities ----------*- C++ -*-===//
//
// This file defines utility classes for Hindley-Milner type inference.
// These utilities are used by both the TypeInference pass and the
// AST-to-Polang conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TRANSFORMS_TYPEINFERENCEUTILS_H
#define POLANG_TRANSFORMS_TYPEINFERENCEUTILS_H

#include "polang/Dialect/PolangTypes.h"

#include "mlir/IR/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace polang {

//===----------------------------------------------------------------------===//
// Helper functions for type variable handling
//===----------------------------------------------------------------------===//

/// Check if a type contains any type variables
inline bool containsTypeVar(mlir::Type type) {
  return mlir::isa<TypeVarType>(type);
}

/// Apply default type for a type variable based on its kind.
/// Returns the resolved type (either the original if not a type var,
/// or the default type based on the kind).
inline mlir::Type applyTypeVarDefault(mlir::Type type, mlir::MLIRContext* ctx) {
  if (auto typeVar = mlir::dyn_cast<TypeVarType>(type)) {
    switch (typeVar.getKind()) {
    case TypeVarKind::Integer:
      return IntegerType::get(ctx, 64, Signedness::Signed);
    case TypeVarKind::Float:
      return FloatType::get(ctx, 64);
    case TypeVarKind::Any:
      // Leave as type var - may be polymorphic
      return type;
    }
  }
  return type;
}

//===----------------------------------------------------------------------===//
// Substitution - Maps type variables to types
//===----------------------------------------------------------------------===//

/// Substitution maps type variable IDs to concrete types.
/// Used during unification to track resolved type bindings.
class Substitution {
public:
  /// Bind a type variable to a type
  void bind(uint64_t var, mlir::Type type) { bindings_[var] = type; }

  /// Look up the binding for a type variable
  /// Returns null Type if not found
  [[nodiscard]] mlir::Type lookup(uint64_t var) const {
    auto it = bindings_.find(var);
    return it != bindings_.end() ? it->second : mlir::Type();
  }

  /// Check if a type variable has a binding
  [[nodiscard]] bool contains(uint64_t var) const {
    return bindings_.contains(var);
  }

  /// Apply substitution to a type, recursively resolving type variables
  [[nodiscard]] mlir::Type apply(mlir::Type type) const {
    if (auto typeVar = mlir::dyn_cast<TypeVarType>(type)) {
      mlir::Type bound = lookup(typeVar.getId());
      if (bound) {
        // Recursively apply in case bound type contains more type vars
        return apply(bound);
      }
      return type;
    }
    // For now, only type variables can be substituted.
    // Function types will need recursive substitution (see issue #9).
    return type;
  }

  /// Compose two substitutions: (this . other)(t) = this(other(t))
  [[nodiscard]] Substitution compose(const Substitution& other) const {
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

  /// Get all bindings for iteration
  [[nodiscard]] const llvm::DenseMap<uint64_t, mlir::Type>& getBindings() const {
    return bindings_;
  }

  /// Debug dump of all bindings
  void dump() const {
    for (const auto& [var, type] : bindings_) {
      llvm::errs() << "  typevar<" << var << "> = " << type << "\n";
    }
  }

private:
  llvm::DenseMap<uint64_t, mlir::Type> bindings_;
};

//===----------------------------------------------------------------------===//
// Unifier - Implements the unification algorithm
//===----------------------------------------------------------------------===//

/// Unifier implements Robinson's unification algorithm for type inference.
/// It attempts to find a substitution that makes two types equal.
class Unifier {
public:
  /// Unify two types, updating the substitution on success.
  /// Returns true if unification succeeds, false otherwise.
  bool unify(mlir::Type t1, mlir::Type t2, Substitution& subst) {
    // Apply current substitution first
    mlir::Type s1 = subst.apply(t1);
    mlir::Type s2 = subst.apply(t2);

    // Same type - trivially unifiable
    if (s1 == s2) {
      return true;
    }

    // Left is type variable
    if (auto var1 = mlir::dyn_cast<TypeVarType>(s1)) {
      return unifyVar(var1.getId(), s2, subst);
    }

    // Right is type variable
    if (auto var2 = mlir::dyn_cast<TypeVarType>(s2)) {
      return unifyVar(var2.getId(), s1, subst);
    }

    // Both are concrete types but different - cannot unify
    return false;
  }

private:
  /// Check if type variable occurs in type (prevents infinite types)
  [[nodiscard]] bool occursIn(uint64_t var, mlir::Type type) const {
    if (auto typeVar = mlir::dyn_cast<TypeVarType>(type)) {
      return typeVar.getId() == var;
    }
    // Function types will need recursive occurs check (see issue #9).
    return false;
  }

  /// Unify a type variable with a type
  bool unifyVar(uint64_t var, mlir::Type type, Substitution& subst) {
    // Occurs check - prevent infinite types
    if (occursIn(var, type)) {
      return false;
    }
    subst.bind(var, type);
    return true;
  }
};

} // namespace polang

#endif // POLANG_TRANSFORMS_TYPEINFERENCEUTILS_H
