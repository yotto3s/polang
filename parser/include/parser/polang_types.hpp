#ifndef POLANG_TYPES_HPP
#define POLANG_TYPES_HPP

#include <optional>
#include <string>

namespace polang {

/// Enumeration of Polang's built-in type kinds.
enum class TypeKind { Int, Double, Bool, Function, TypeVar, Unknown };

/// Type name constants to avoid magic strings throughout the codebase.
/// Use these constants instead of string literals like "int", "double", etc.
struct TypeNames {
  static constexpr const char* INT = "int";
  static constexpr const char* DOUBLE = "double";
  static constexpr const char* BOOL = "bool";
  static constexpr const char* FUNCTION = "function";
  static constexpr const char* TYPEVAR = "typevar";
  static constexpr const char* UNKNOWN = "unknown";
};

/// Convert a TypeKind enum to its string representation.
inline const char* typeKindToString(TypeKind kind) noexcept {
  switch (kind) {
  case TypeKind::Int:
    return TypeNames::INT;
  case TypeKind::Double:
    return TypeNames::DOUBLE;
  case TypeKind::Bool:
    return TypeNames::BOOL;
  case TypeKind::Function:
    return TypeNames::FUNCTION;
  case TypeKind::TypeVar:
    return TypeNames::TYPEVAR;
  case TypeKind::Unknown:
    return TypeNames::UNKNOWN;
  }
  return TypeNames::UNKNOWN;
}

/// Parse a type name string into a TypeKind enum.
/// Returns std::nullopt if the string is not a recognized type name.
inline std::optional<TypeKind> parseTypeName(const std::string& name) noexcept {
  if (name == TypeNames::INT)
    return TypeKind::Int;
  if (name == TypeNames::DOUBLE)
    return TypeKind::Double;
  if (name == TypeNames::BOOL)
    return TypeKind::Bool;
  if (name == TypeNames::FUNCTION)
    return TypeKind::Function;
  if (name == TypeNames::TYPEVAR)
    return TypeKind::TypeVar;
  if (name == TypeNames::UNKNOWN)
    return TypeKind::Unknown;
  return std::nullopt;
}

/// Check if a type name represents a numeric type (int or double).
inline bool isNumericType(const std::string& typeName) noexcept {
  return typeName == TypeNames::INT || typeName == TypeNames::DOUBLE;
}

/// Check if a type name represents a boolean type.
inline bool isBooleanType(const std::string& typeName) noexcept {
  return typeName == TypeNames::BOOL;
}

/// Check if a type name represents an unknown/error type.
inline bool isUnknownType(const std::string& typeName) noexcept {
  return typeName == TypeNames::UNKNOWN;
}

} // namespace polang

#endif // POLANG_TYPES_HPP
