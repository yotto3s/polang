//===- PolangTypeConverter.cpp - Type conversion for Polang -----*- C++ -*-===//
//
// This file implements the PolangTypeConverter class.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/PolangTypeConverter.h"
#include "polang/Dialect/PolangTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

#pragma GCC diagnostic pop

// clang-format off
#include "parser/node.hpp"
#include "parser.hpp"  // Must be after node.hpp for bison union types
// clang-format on
#include "parser/polang_types.hpp"

using namespace mlir;
using namespace polang;

PolangTypeConverter::PolangTypeConverter(MLIRContext* ctx) : context(ctx) {}

Type PolangTypeConverter::getTypeParamType(llvm::StringRef name) {
  // Strip the leading quote if present (e.g., "'a" -> "a")
  if (name.starts_with("'")) {
    name = name.drop_front(1);
  }
  return TypeParamType::get(context, name);
}

Type PolangTypeConverter::getTypeOrFresh(const NTypeSpec* typeAnnotation) {
  if (typeAnnotation != nullptr) {
    Type ty = getPolangType(*typeAnnotation);
    if (!ty) {
      return nullptr; // Propagate error
    }
    return ty;
  }
  // No annotation - use default type (AST-level inference handles resolution)
  return getDefaultType();
}

Type PolangTypeConverter::getPolangType(const NTypeSpec& typeSpec) {
  // Handle NNamedType
  const auto* named = dynamic_cast<const NNamedType*>(&typeSpec);
  if (named == nullptr) {
    // Unknown type specification - should not happen
    return nullptr;
  }

  const std::string& typeName = named->name;

  // Handle type parameters ('a, 'b, etc.)
  if (typeName.size() >= 2 && typeName[0] == '\'') {
    return getTypeParamType(typeName);
  }

  const TypeMetadata meta = getTypeMetadata(typeName);

  switch (meta.kind) {
  case TypeKind::Integer:
    if (meta.isGeneric) {
      // Generic integer defaults to si64
      return mlir::IntegerType::get(context, DEFAULT_INT_WIDTH,
                                    mlir::IntegerType::Signed);
    }
    return mlir::IntegerType::get(context, meta.width,
                                  meta.isSigned()
                                      ? mlir::IntegerType::Signed
                                      : mlir::IntegerType::Unsigned);

  case TypeKind::Float:
    if (meta.isGeneric || meta.width == 64) {
      // Generic float defaults to f64
      return Float64Type::get(context);
    }
    return Float32Type::get(context);

  case TypeKind::Bool:
    return mlir::IntegerType::get(context, 1);

  case TypeKind::Index:
    return mlir::IndexType::get(context);

  case TypeKind::Unit:
  case TypeKind::TypeVar:
  case TypeKind::Function:
  case TypeKind::Unknown:
    // Default to si64 for unresolved/unknown types
    return getDefaultType();
  }

  // Unreachable, but needed for compiler
  return getDefaultType();
}

Type PolangTypeConverter::getDefaultType() {
  return mlir::IntegerType::get(context, DEFAULT_INT_WIDTH,
                                mlir::IntegerType::Signed);
}

Type PolangTypeConverter::convertPolangType(Type polangType) {
  return polangType;
}
