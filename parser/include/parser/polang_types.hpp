#ifndef POLANG_TYPES_HPP
#define POLANG_TYPES_HPP

#include <optional>
#include <string>

namespace polang {

/// Enumeration of Polang's built-in type kinds.
enum class TypeKind {
  Integer,
  Float,
  Bool,
  Function,
  TypeVar,
  MutableRef,
  ImmutableRef,
  Unknown
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
  // Reference type prefixes
  static constexpr const char* MUT_PREFIX = "mut ";
  static constexpr const char* REF_PREFIX = "ref ";
};

/// Parse a type name string into a TypeKind enum.
/// Returns std::nullopt if the string is not a recognized type name.
[[nodiscard]] inline std::optional<TypeKind>
parseTypeName(const std::string& name) noexcept {
  // Check for reference types first (before other checks)
  if (name.rfind(TypeNames::MUT_PREFIX, 0) == 0) {
    return TypeKind::MutableRef;
  }
  if (name.rfind(TypeNames::REF_PREFIX, 0) == 0) {
    return TypeKind::ImmutableRef;
  }
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

/// Check if a type contains a generic type (including reference types).
/// For example, "ref {int}" and "mut {int}" contain generic types.
[[nodiscard]] inline bool
containsGenericType(const std::string& typeName) noexcept {
  if (isGenericType(typeName)) {
    return true;
  }
  // Check if it's a reference type with a generic referent
  if (typeName.rfind(TypeNames::MUT_PREFIX, 0) == 0) {
    return containsGenericType(typeName.substr(4)); // Skip "mut "
  }
  if (typeName.rfind(TypeNames::REF_PREFIX, 0) == 0) {
    return containsGenericType(typeName.substr(4)); // Skip "ref "
  }
  return false;
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
/// Handles reference types containing generic types.
/// For example, "ref {int}" becomes "ref i64".
[[nodiscard]] inline std::string
resolveAllGenericsToDefault(const std::string& type) noexcept {
  if (isGenericType(type)) {
    return resolveGenericToDefault(type);
  }
  // Handle reference types
  if (type.rfind(TypeNames::MUT_PREFIX, 0) == 0) {
    return std::string(TypeNames::MUT_PREFIX) +
           resolveAllGenericsToDefault(type.substr(4));
  }
  if (type.rfind(TypeNames::REF_PREFIX, 0) == 0) {
    return std::string(TypeNames::REF_PREFIX) +
           resolveAllGenericsToDefault(type.substr(4));
  }
  return type;
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
  case TypeKind::MutableRef:
    return "mutref";
  case TypeKind::ImmutableRef:
    return "ref";
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

//===----------------------------------------------------------------------===//
// Reference Type Helpers
//===----------------------------------------------------------------------===//

/// Check if a type name represents a mutable reference type.
[[nodiscard]] inline bool
isMutableRefType(const std::string& typeName) noexcept {
  return typeName.rfind(TypeNames::MUT_PREFIX, 0) == 0;
}

/// Check if a type name represents an immutable reference type.
[[nodiscard]] inline bool
isImmutableRefType(const std::string& typeName) noexcept {
  return typeName.rfind(TypeNames::REF_PREFIX, 0) == 0;
}

/// Check if a type name represents any reference type (mutable or immutable).
[[nodiscard]] inline bool
isReferenceType(const std::string& typeName) noexcept {
  return isMutableRefType(typeName) || isImmutableRefType(typeName);
}

/// Get the referent type from a reference type.
/// For "mut i64" returns "i64", for "ref i64" returns "i64".
/// Returns the input unchanged if it's not a reference type.
[[nodiscard]] inline std::string
getReferentType(const std::string& refType) noexcept {
  if (isMutableRefType(refType)) {
    return refType.substr(4); // Skip "mut "
  }
  if (isImmutableRefType(refType)) {
    return refType.substr(4); // Skip "ref "
  }
  return refType;
}

/// Create a mutable reference type from a base type.
/// For "i64" returns "mut i64".
[[nodiscard]] inline std::string
makeMutableRefType(const std::string& baseType) noexcept {
  return std::string(TypeNames::MUT_PREFIX) + baseType;
}

/// Create an immutable reference type from a base type.
/// For "i64" returns "ref i64".
[[nodiscard]] inline std::string
makeImmutableRefType(const std::string& baseType) noexcept {
  return std::string(TypeNames::REF_PREFIX) + baseType;
}

} // namespace polang

#endif // POLANG_TYPES_HPP
