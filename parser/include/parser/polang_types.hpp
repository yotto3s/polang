#ifndef POLANG_TYPES_HPP
#define POLANG_TYPES_HPP

#include <optional>
#include <string>

namespace polang {

/// Enumeration of Polang's built-in type kinds.
enum class TypeKind { Integer, Float, Bool, Function, TypeVar, Unknown };

/// Type name constants to avoid magic strings throughout the codebase.
/// Use these constants instead of string literals like "i32", "f64", etc.
struct TypeNames {
  // Signed integers
  static constexpr const char* I8 = "i8";
  static constexpr const char* I16 = "i16";
  static constexpr const char* I32 = "i32";
  static constexpr const char* I64 = "i64";
  // Unsigned integers
  static constexpr const char* U8 = "u8";
  static constexpr const char* U16 = "u16";
  static constexpr const char* U32 = "u32";
  static constexpr const char* U64 = "u64";
  // Floats
  static constexpr const char* F32 = "f32";
  static constexpr const char* F64 = "f64";
  // Other types
  static constexpr const char* BOOL = "bool";
  static constexpr const char* FUNCTION = "function";
  static constexpr const char* TYPEVAR = "typevar";
  static constexpr const char* UNKNOWN = "unknown";
};

/// Parse a type name string into a TypeKind enum.
/// Returns std::nullopt if the string is not a recognized type name.
[[nodiscard]] inline std::optional<TypeKind>
parseTypeName(const std::string& name) noexcept {
  // Signed integers
  if (name == TypeNames::I8 || name == TypeNames::I16 ||
      name == TypeNames::I32 || name == TypeNames::I64) {
    return TypeKind::Integer;
  }
  // Unsigned integers
  if (name == TypeNames::U8 || name == TypeNames::U16 ||
      name == TypeNames::U32 || name == TypeNames::U64) {
    return TypeKind::Integer;
  }
  // Floats
  if (name == TypeNames::F32 || name == TypeNames::F64) {
    return TypeKind::Float;
  }
  // Other types
  if (name == TypeNames::BOOL || name == "bool") {
    return TypeKind::Bool;
  }
  if (name == TypeNames::FUNCTION || name == "function") {
    return TypeKind::Function;
  }
  if (name == TypeNames::TYPEVAR || name == "typevar") {
    return TypeKind::TypeVar;
  }
  if (name == TypeNames::UNKNOWN || name == "unknown") {
    return TypeKind::Unknown;
  }
  // Also accept kind names for round-trip testing
  if (name == "integer") {
    return TypeKind::Integer;
  }
  if (name == "float") {
    return TypeKind::Float;
  }
  return std::nullopt;
}

/// Check if a type name represents an integer type (signed or unsigned).
[[nodiscard]] inline bool isIntegerType(const std::string& typeName) noexcept {
  return typeName == TypeNames::I8 || typeName == TypeNames::I16 ||
         typeName == TypeNames::I32 || typeName == TypeNames::I64 ||
         typeName == TypeNames::U8 || typeName == TypeNames::U16 ||
         typeName == TypeNames::U32 || typeName == TypeNames::U64;
}

/// Check if a type name represents a signed integer type.
[[nodiscard]] inline bool
isSignedIntegerType(const std::string& typeName) noexcept {
  return typeName == TypeNames::I8 || typeName == TypeNames::I16 ||
         typeName == TypeNames::I32 || typeName == TypeNames::I64;
}

/// Check if a type name represents an unsigned integer type.
[[nodiscard]] inline bool
isUnsignedIntegerType(const std::string& typeName) noexcept {
  return typeName == TypeNames::U8 || typeName == TypeNames::U16 ||
         typeName == TypeNames::U32 || typeName == TypeNames::U64;
}

/// Check if a type name represents a floating-point type.
[[nodiscard]] inline bool isFloatType(const std::string& typeName) noexcept {
  return typeName == TypeNames::F32 || typeName == TypeNames::F64;
}

/// Check if a type name represents a numeric type (integer or float).
[[nodiscard]] inline bool isNumericType(const std::string& typeName) noexcept {
  return isIntegerType(typeName) || isFloatType(typeName);
}

/// Check if a type name represents a boolean type.
[[nodiscard]] inline bool isBooleanType(const std::string& typeName) noexcept {
  return typeName == TypeNames::BOOL;
}

/// Check if a type name represents an unknown/error type.
[[nodiscard]] inline bool isUnknownType(const std::string& typeName) noexcept {
  return typeName == TypeNames::UNKNOWN;
}

/// Convert a TypeKind to its string representation.
/// Returns the kind name (e.g., "integer", "float", "bool").
[[nodiscard]] inline const char* typeKindToString(TypeKind kind) noexcept {
  switch (kind) {
  case TypeKind::Integer:
    return "integer";
  case TypeKind::Float:
    return "float";
  case TypeKind::Bool:
    return "bool";
  case TypeKind::Function:
    return "function";
  case TypeKind::TypeVar:
    return "typevar";
  case TypeKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

/// Get the bit width of an integer type. Returns 0 for non-integer types.
[[nodiscard]] inline unsigned
getIntegerWidth(const std::string& typeName) noexcept {
  if (typeName == TypeNames::I8 || typeName == TypeNames::U8) {
    return 8;
  }
  if (typeName == TypeNames::I16 || typeName == TypeNames::U16) {
    return 16;
  }
  if (typeName == TypeNames::I32 || typeName == TypeNames::U32) {
    return 32;
  }
  if (typeName == TypeNames::I64 || typeName == TypeNames::U64) {
    return 64;
  }
  return 0;
}

/// Get the bit width of a float type. Returns 0 for non-float types.
[[nodiscard]] inline unsigned
getFloatWidth(const std::string& typeName) noexcept {
  if (typeName == TypeNames::F32) {
    return 32;
  }
  if (typeName == TypeNames::F64) {
    return 64;
  }
  return 0;
}

} // namespace polang

#endif // POLANG_TYPES_HPP
