#ifndef POLANG_TYPE_INFERENCE_HPP
#define POLANG_TYPE_INFERENCE_HPP

#include <map>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace polang {

/// Trait bounds for type parameters, inferred by the AST type checker.
enum class TraitBound { Numeric, Integer, Float };

/// Convert a TraitBound to its string representation.
[[nodiscard]] const char* traitBoundToString(TraitBound bound) noexcept;

/// Parse a string to a TraitBound. Returns nullopt for unknown strings.
[[nodiscard]] std::optional<TraitBound>
stringToTraitBound(const std::string& str) noexcept;

/// Check if a string represents a unification variable ($0, $1, ...).
[[nodiscard]] bool isUnificationVar(const std::string& name) noexcept;

/// Generate a fresh unification variable ($0, $1, ...).
[[nodiscard]] std::string freshUnificationVar();

/// Reset the unification variable counter (for testing).
void resetUnificationVarCounter() noexcept;

/// Check if a string represents a type parameter ('a, 'b, ...).
[[nodiscard]] bool isTypeParameter(const std::string& name) noexcept;

/// Substitution mapping from unification variables to types.
/// Supports transitive resolution: if $0 -> $1 and $1 -> i64, then
/// apply($0) returns i64.
class Substitution {
public:
  /// Bind a unification variable to a type.
  void bind(const std::string& var, const std::string& type);

  /// Apply the substitution to a type, resolving transitively.
  [[nodiscard]] std::string apply(const std::string& type) const;

  /// Compose this substitution with another: apply other to all existing
  /// bindings, then add any new bindings from other.
  [[nodiscard]] Substitution compose(const Substitution& other) const;

  /// Get the raw bindings map (for inspection).
  [[nodiscard]] const std::map<std::string, std::string>&
  getBindings() const noexcept {
    return bindings;
  }

private:
  std::map<std::string, std::string> bindings;
};

/// Unifier implementing Robinson's unification algorithm.
/// Handles unification variables ($N), generic types ({int}, {float}),
/// type parameters ('a), and concrete types.
class Unifier {
public:
  /// Attempt to unify two types. Returns true on success and updates the
  /// substitution. Returns false if the types cannot be unified.
  bool unify(const std::string& t1, const std::string& t2, Substitution& subst);
};

/// Tracks trait bounds accumulated during type inference.
/// Maps unification variables to their accumulated bounds.
class TraitConstraints {
public:
  /// Add a trait bound to a unification variable.
  void addBound(const std::string& var, TraitBound bound);

  /// Get the trait bounds for a unification variable.
  [[nodiscard]] std::set<TraitBound> getBounds(const std::string& var) const;

  /// Check if a concrete type satisfies a set of trait bounds.
  [[nodiscard]] static bool
  satisfies(const std::string& concreteType,
            const std::set<TraitBound>& bounds) noexcept;

private:
  std::map<std::string, std::set<TraitBound>> bounds;
};

/// Monomorphic function signature: concrete param types and return type.
struct MonoSignature {
  std::vector<std::string> paramTypes;
  std::string returnType;
};

/// Polymorphic function signature: forall 'a:'Numeric, 'b. ('a, 'b) -> 'a
struct PolymorphicSignature {
  std::vector<std::string> typeParams;                     // ["'a", "'b"]
  std::map<std::string, std::set<TraitBound>> paramBounds; // {"'a": {Numeric}}
  std::vector<std::string> paramTypes; // param types (may reference params)
  std::string returnType;
};

/// A function signature is either monomorphic or polymorphic.
using FunctionSignature = std::variant<MonoSignature, PolymorphicSignature>;

} // namespace polang

#endif // POLANG_TYPE_INFERENCE_HPP
