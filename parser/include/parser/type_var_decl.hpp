#ifndef POLANG_TYPE_VAR_DECL_HPP
#define POLANG_TYPE_VAR_DECL_HPP

#include <string>
#include <utility>

// Declaration of a type variable in a forall quantifier
// e.g., 'a or 'a:Numeric
struct TypeVarDecl {
  std::string name;  // e.g., "'a"
  std::string bound; // e.g., "Numeric" (empty if unconstrained)
  TypeVarDecl() = default;
  TypeVarDecl(std::string name, std::string bound = "")
      : name(std::move(name)), bound(std::move(bound)) {}
};

#endif // POLANG_TYPE_VAR_DECL_HPP
