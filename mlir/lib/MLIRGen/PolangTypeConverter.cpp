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
      // Generic integer defaults to i64
      return polang::IntegerType::get(context, DEFAULT_INT_WIDTH,
                                      Signedness::Signed);
    }
    return polang::IntegerType::get(context, meta.width,
                                    meta.isSigned() ? Signedness::Signed
                                                    : Signedness::Unsigned);

  case TypeKind::Float:
    if (meta.isGeneric) {
      // Generic float defaults to f64
      return polang::FloatType::get(context, DEFAULT_FLOAT_WIDTH);
    }
    return polang::FloatType::get(context, meta.width);

  case TypeKind::Bool:
    return BoolType::get(context);

  case TypeKind::TypeVar:
  case TypeKind::Function:
  case TypeKind::Unknown:
    // Default to i64 for unresolved/unknown types
    return getDefaultType();
  }

  // Unreachable, but needed for compiler
  return getDefaultType();
}

Type PolangTypeConverter::getDefaultType() {
  return polang::IntegerType::get(context, DEFAULT_INT_WIDTH,
                                  Signedness::Signed);
}

Type PolangTypeConverter::convertPolangType(Type polangType) {
  if (auto intType = dyn_cast<polang::IntegerType>(polangType)) {
    return mlir::IntegerType::get(context, intType.getWidth());
  }
  if (auto floatType = dyn_cast<polang::FloatType>(polangType)) {
    if (floatType.getWidth() == 32) {
      return Float32Type::get(context);
    }
    return Float64Type::get(context);
  }
  if (isa<BoolType>(polangType)) {
    return mlir::IntegerType::get(context, 1);
  }
  return mlir::IntegerType::get(context, DEFAULT_INT_WIDTH);
}
