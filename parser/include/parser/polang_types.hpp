#ifndef POLANG_TYPES_HPP
#define POLANG_TYPES_HPP

#include <optional>
#include <string>

namespace polang {

/// Default bit widths for types when not explicitly specified.
constexpr unsigned DEFAULT_INT_WIDTH = 64;
constexpr unsigned DEFAULT_FLOAT_WIDTH = 64;

/// Enumeration of Polang's built-in type kinds.
enum class TypeKind { Integer, Float, Bool, Function, TypeVar, Unknown };

/// Signedness for integer types.
enum class TypeSignedness { Signed, Unsigned };

/// Metadata about a Polang type, capturing width, signedness, and kind.
/// Used to centralize type information and avoid repeated string parsing.
struct TypeMetadata {
  TypeKind kind = TypeKind::Unknown;
  unsigned width = 0;
  TypeSignedness signedness = TypeSignedness::Signed;
  bool isGeneric = false;

  [[nodiscard]] constexpr bool isInteger() const noexcept {
    return kind == TypeKind::Integer;
  }
  [[nodiscard]] constexpr bool isFloat() const noexcept {
    return kind == TypeKind::Float;
  }
  [[nodiscard]] constexpr bool isBool() const noexcept {
    return kind == TypeKind::Bool;
  }
  [[nodiscard]] constexpr bool isSigned() const noexcept {
    return signedness == TypeSignedness::Signed;
  }
  [[nodiscard]] constexpr bool isUnsigned() const noexcept {
    return signedness == TypeSignedness::Unsigned;
  }
  [[nodiscard]] constexpr bool isNumeric() const noexcept {
    return isInteger() || isFloat();
  }
};

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
  // Generic types (for literals before type inference)
  static constexpr const char* GENERIC_INT = "{int}";
  static constexpr const char* GENERIC_FLOAT = "{float}";
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
  // Generic integer (unresolved literal)
  if (name == TypeNames::GENERIC_INT) {
    return TypeKind::Integer;
  }
  // Floats
  if (name == TypeNames::F32 || name == TypeNames::F64) {
    return TypeKind::Float;
  }
  // Generic float (unresolved literal)
  if (name == TypeNames::GENERIC_FLOAT) {
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

/// Check if a type name represents a generic (unresolved) integer type.
[[nodiscard]] inline bool
isGenericIntegerType(const std::string& typeName) noexcept {
  return typeName == TypeNames::GENERIC_INT;
}

/// Check if a type name represents a generic (unresolved) float type.
[[nodiscard]] inline bool
isGenericFloatType(const std::string& typeName) noexcept {
  return typeName == TypeNames::GENERIC_FLOAT;
}

/// Check if a type name represents any generic (unresolved) type.
[[nodiscard]] inline bool isGenericType(const std::string& typeName) noexcept {
  return isGenericIntegerType(typeName) || isGenericFloatType(typeName);
}

/// Check if a type contains a generic type.
[[nodiscard]] inline bool
containsGenericType(const std::string& typeName) noexcept {
  return isGenericType(typeName);
}

/// Check if two types are compatible for assignment/operations.
/// Generic types are compatible with their concrete counterparts.
[[nodiscard]] inline bool areTypesCompatible(const std::string& t1,
                                             const std::string& t2) noexcept {
  // Exact match
  if (t1 == t2) {
    return true;
  }
  // Generic int is compatible with any concrete integer type
  if (isGenericIntegerType(t1) && isIntegerType(t2)) {
    return true;
  }
  if (isGenericIntegerType(t2) && isIntegerType(t1)) {
    return true;
  }
  // Generic float is compatible with any concrete float type
  if (isGenericFloatType(t1) && isFloatType(t2)) {
    return true;
  }
  if (isGenericFloatType(t2) && isFloatType(t1)) {
    return true;
  }
  // Two generic ints are compatible
  if (isGenericIntegerType(t1) && isGenericIntegerType(t2)) {
    return true;
  }
  // Two generic floats are compatible
  if (isGenericFloatType(t1) && isGenericFloatType(t2)) {
    return true;
  }
  return false;
}

/// Resolve a generic type to a concrete type given a context type.
/// If the type is not generic, returns it unchanged.
/// If context is also generic, returns the default type.
[[nodiscard]] inline std::string
resolveGenericType(const std::string& type,
                   const std::string& contextType) noexcept {
  if (isGenericIntegerType(type)) {
    if (isIntegerType(contextType)) {
      return contextType;
    }
    // Default to i64
    return TypeNames::I64;
  }
  if (isGenericFloatType(type)) {
    if (isFloatType(contextType)) {
      return contextType;
    }
    // Default to f64
    return TypeNames::F64;
  }
  return type;
}

/// Resolve a generic type to its default concrete type.
[[nodiscard]] inline std::string
resolveGenericToDefault(const std::string& type) noexcept {
  if (isGenericIntegerType(type)) {
    return TypeNames::I64;
  }
  if (isGenericFloatType(type)) {
    return TypeNames::F64;
  }
  return type;
}

/// Resolve all generic types within a type to their defaults.
[[nodiscard]] inline std::string
resolveAllGenericsToDefault(const std::string& type) noexcept {
  return resolveGenericToDefault(type);
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

/// Get comprehensive metadata about a type from its name.
/// Returns metadata with kind=Unknown for unrecognized types.
[[nodiscard]] inline TypeMetadata
getTypeMetadata(const std::string& typeName) noexcept {
  TypeMetadata meta;

  // Signed integers
  if (typeName == TypeNames::I8) {
    meta.kind = TypeKind::Integer;
    meta.width = 8;
    meta.signedness = TypeSignedness::Signed;
    return meta;
  }
  if (typeName == TypeNames::I16) {
    meta.kind = TypeKind::Integer;
    meta.width = 16;
    meta.signedness = TypeSignedness::Signed;
    return meta;
  }
  if (typeName == TypeNames::I32) {
    meta.kind = TypeKind::Integer;
    meta.width = 32;
    meta.signedness = TypeSignedness::Signed;
    return meta;
  }
  if (typeName == TypeNames::I64) {
    meta.kind = TypeKind::Integer;
    meta.width = 64;
    meta.signedness = TypeSignedness::Signed;
    return meta;
  }

  // Unsigned integers
  if (typeName == TypeNames::U8) {
    meta.kind = TypeKind::Integer;
    meta.width = 8;
    meta.signedness = TypeSignedness::Unsigned;
    return meta;
  }
  if (typeName == TypeNames::U16) {
    meta.kind = TypeKind::Integer;
    meta.width = 16;
    meta.signedness = TypeSignedness::Unsigned;
    return meta;
  }
  if (typeName == TypeNames::U32) {
    meta.kind = TypeKind::Integer;
    meta.width = 32;
    meta.signedness = TypeSignedness::Unsigned;
    return meta;
  }
  if (typeName == TypeNames::U64) {
    meta.kind = TypeKind::Integer;
    meta.width = 64;
    meta.signedness = TypeSignedness::Unsigned;
    return meta;
  }

  // Generic integer (unresolved literal)
  if (typeName == TypeNames::GENERIC_INT) {
    meta.kind = TypeKind::Integer;
    meta.width = DEFAULT_INT_WIDTH;
    meta.signedness = TypeSignedness::Signed;
    meta.isGeneric = true;
    return meta;
  }

  // Floats
  if (typeName == TypeNames::F32) {
    meta.kind = TypeKind::Float;
    meta.width = 32;
    return meta;
  }
  if (typeName == TypeNames::F64) {
    meta.kind = TypeKind::Float;
    meta.width = 64;
    return meta;
  }

  // Generic float (unresolved literal)
  if (typeName == TypeNames::GENERIC_FLOAT) {
    meta.kind = TypeKind::Float;
    meta.width = DEFAULT_FLOAT_WIDTH;
    meta.isGeneric = true;
    return meta;
  }

  // Bool
  if (typeName == TypeNames::BOOL) {
    meta.kind = TypeKind::Bool;
    meta.width = 1;
    return meta;
  }

  // Function, TypeVar, Unknown
  if (typeName == TypeNames::FUNCTION) {
    meta.kind = TypeKind::Function;
    return meta;
  }
  if (typeName == TypeNames::TYPEVAR) {
    meta.kind = TypeKind::TypeVar;
    return meta;
  }

  // Unknown type
  meta.kind = TypeKind::Unknown;
  return meta;
}

/// Get the bit width of an integer type. Returns 0 for non-integer types.
[[nodiscard]] inline unsigned
getIntegerWidth(const std::string& typeName) noexcept {
  const TypeMetadata meta = getTypeMetadata(typeName);
  return meta.isInteger() ? meta.width : 0;
}

/// Get the bit width of a float type. Returns 0 for non-float types.
[[nodiscard]] inline unsigned
getFloatWidth(const std::string& typeName) noexcept {
  const TypeMetadata meta = getTypeMetadata(typeName);
  return meta.isFloat() ? meta.width : 0;
}

} // namespace polang

#endif // POLANG_TYPES_HPP
