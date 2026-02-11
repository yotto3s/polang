#include "parser/type_inference.hpp"
#include "parser/polang_types.hpp"

#include <algorithm>

namespace polang {

namespace {
thread_local int unificationVarCounter = 0;
} // namespace

const char* traitBoundToString(TraitBound bound) noexcept {
  switch (bound) {
  case TraitBound::Numeric:
    return "Numeric";
  case TraitBound::Integer:
    return "Integer";
  case TraitBound::Float:
    return "Float";
  }
  return "Numeric";
}

std::optional<TraitBound> stringToTraitBound(const std::string& str) noexcept {
  if (str == "Numeric") {
    return TraitBound::Numeric;
  }
  if (str == "Integer") {
    return TraitBound::Integer;
  }
  if (str == "Float") {
    return TraitBound::Float;
  }
  return std::nullopt;
}

bool isUnificationVar(const std::string& name) noexcept {
  if (name.size() < 2 || name[0] != '$') {
    return false;
  }
  for (size_t i = 1; i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') {
      return false;
    }
  }
  return true;
}

std::string freshUnificationVar() {
  return "$" + std::to_string(unificationVarCounter++);
}

void resetUnificationVarCounter() noexcept { unificationVarCounter = 0; }

bool isTypeParameter(const std::string& name) noexcept {
  return name.size() >= 2 && name[0] == '\'';
}

// ============== Substitution ==============

void Substitution::bind(const std::string& var, const std::string& type) {
  bindings[var] = type;
}

std::string Substitution::apply(const std::string& type) const {
  std::string current = type;
  // Follow the chain of bindings (with loop protection)
  static constexpr int MAX_SUBSTITUTION_DEPTH = 100;
  int limit = MAX_SUBSTITUTION_DEPTH;
  while (limit-- > 0) {
    auto it = bindings.find(current);
    if (it == bindings.end()) {
      return current;
    }
    if (it->second == current) {
      return current;
    }
    current = it->second;
  }
  return current;
}

Substitution Substitution::compose(const Substitution& other) const {
  Substitution result;
  // Apply other to all existing bindings in this substitution
  for (const auto& [var, type] : bindings) {
    result.bindings[var] = other.apply(type);
  }
  // Add bindings from other that aren't already in result
  for (const auto& [var, type] : other.bindings) {
    if (result.bindings.find(var) == result.bindings.end()) {
      result.bindings[var] = type;
    }
  }
  return result;
}

// ============== Unifier ==============

bool Unifier::unify(const std::string& t1, const std::string& t2,
                    Substitution& subst) {
  // Apply current substitution first
  std::string s1 = subst.apply(t1);
  std::string s2 = subst.apply(t2);

  // Same type - trivially unified
  if (s1 == s2) {
    return true;
  }

  // If either is a unification variable, bind it
  if (isUnificationVar(s1)) {
    subst.bind(s1, s2);
    return true;
  }
  if (isUnificationVar(s2)) {
    subst.bind(s2, s1);
    return true;
  }

  // Handle generic types: {int} is compatible with concrete integer types
  if (isGenericIntegerType(s1) && (isIntegerType(s2) || isIndexType(s2))) {
    return true;
  }
  if (isGenericIntegerType(s2) && (isIntegerType(s1) || isIndexType(s1))) {
    return true;
  }
  if (isGenericFloatType(s1) && isFloatType(s2)) {
    return true;
  }
  if (isGenericFloatType(s2) && isFloatType(s1)) {
    return true;
  }
  // Two generic ints or two generic floats are compatible
  if (isGenericIntegerType(s1) && isGenericIntegerType(s2)) {
    return true;
  }
  if (isGenericFloatType(s1) && isGenericFloatType(s2)) {
    return true;
  }

  // Otherwise, types are incompatible
  return false;
}

// ============== TraitConstraints ==============

void TraitConstraints::addBound(const std::string& var, TraitBound bound) {
  bounds[var].insert(bound);
}

std::set<TraitBound> TraitConstraints::getBounds(const std::string& var) const {
  auto it = bounds.find(var);
  if (it != bounds.end()) {
    return it->second;
  }
  return {};
}

bool TraitConstraints::satisfies(const std::string& concreteType,
                                 const std::set<TraitBound>& boundsSet) {
  auto& registry = getTraitRegistry();
  return std::all_of(
      boundsSet.begin(), boundsSet.end(), [&](const TraitBound& bound) {
        return registry.satisfies(concreteType, traitBoundToString(bound));
      });
}

// ============== TraitRegistry ==============

TraitRegistry::TraitRegistry() {
  // Built-in Numeric trait: all integer and float types
  TraitDefinition numeric;
  numeric.name = "Numeric";
  numeric.methods = {
      {"+", {"'self", "'self"}, "'self"},
      {"-", {"'self", "'self"}, "'self"},
      {"*", {"'self", "'self"}, "'self"},
      {"/", {"'self", "'self"}, "'self"},
  };
  numeric.satisfyingTypes = {
      TypeNames::I8,    TypeNames::I16,   TypeNames::I32, TypeNames::I64,
      TypeNames::U8,    TypeNames::U16,   TypeNames::U32, TypeNames::U64,
      TypeNames::ISIZE, TypeNames::USIZE, TypeNames::F32, TypeNames::F64,
  };
  registerTrait(std::move(numeric));

  // Built-in Integer trait: integer types only
  TraitDefinition integer;
  integer.name = "Integer";
  integer.methods = {
      {"%", {"'self", "'self"}, "'self"},
  };
  integer.satisfyingTypes = {
      TypeNames::I8,    TypeNames::I16,   TypeNames::I32, TypeNames::I64,
      TypeNames::U8,    TypeNames::U16,   TypeNames::U32, TypeNames::U64,
      TypeNames::ISIZE, TypeNames::USIZE,
  };
  registerTrait(std::move(integer));

  // Built-in Float trait: float types only
  TraitDefinition floatTrait;
  floatTrait.name = "Float";
  floatTrait.methods = {};
  floatTrait.satisfyingTypes = {
      TypeNames::F32,
      TypeNames::F64,
  };
  registerTrait(std::move(floatTrait));
}

void TraitRegistry::registerTrait(TraitDefinition def) {
  std::string name = def.name;
  // Build reverse index from method names to trait
  for (const auto& method : def.methods) {
    methodToTrait[method.methodName] = name;
  }
  traits[std::move(name)] = std::move(def);
}

bool TraitRegistry::isKnownTrait(const std::string& name) const {
  return traits.find(name) != traits.end();
}

bool TraitRegistry::satisfies(const std::string& concreteType,
                              const std::string& traitName) const {
  auto it = traits.find(traitName);
  if (it == traits.end()) {
    return false;
  }
  return it->second.satisfyingTypes.count(concreteType) > 0;
}

std::optional<std::string>
TraitRegistry::traitForMethod(const std::string& method) const {
  auto it = methodToTrait.find(method);
  if (it != methodToTrait.end()) {
    return it->second;
  }
  return std::nullopt;
}

const TraitDefinition* TraitRegistry::getTrait(const std::string& name) const {
  auto it = traits.find(name);
  if (it != traits.end()) {
    return &it->second;
  }
  return nullptr;
}

TraitRegistry& getTraitRegistry() {
  static TraitRegistry registry;
  return registry;
}

} // namespace polang
