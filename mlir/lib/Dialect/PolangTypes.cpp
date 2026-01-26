//===- PolangTypes.cpp - Polang type implementation -------------*- C++ -*-===//
//
// This file implements the types for the Polang dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "polang/Dialect/PolangDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace polang;

#define GET_TYPEDEF_CLASSES
#include "polang/Dialect/PolangTypes.cpp.inc"

#pragma GCC diagnostic pop

//===----------------------------------------------------------------------===//
// TypeVarType custom print/parse
//===----------------------------------------------------------------------===//

void TypeVarType::print(AsmPrinter& printer) const {
  printer << "<" << getId();
  if (getKind() != TypeVarKind::Any) {
    printer << ", " << stringifyTypeVarKind(getKind());
  }
  printer << ">";
}

Type TypeVarType::parse(AsmParser& parser) {
  uint64_t id = 0;
  TypeVarKind kind = TypeVarKind::Any;

  if (parser.parseLess() || parser.parseInteger(id)) {
    return {};
  }

  // Check for optional kind
  if (succeeded(parser.parseOptionalComma())) {
    StringRef kindStr;
    if (parser.parseKeyword(&kindStr)) {
      return {};
    }
    auto maybeKind = symbolizeTypeVarKind(kindStr);
    if (!maybeKind) {
      parser.emitError(parser.getCurrentLocation(),
                       "invalid type variable kind: ")
          << kindStr;
      return {};
    }
    kind = *maybeKind;
  }

  if (parser.parseGreater()) {
    return {};
  }

  return TypeVarType::get(parser.getContext(), id, kind);
}

//===----------------------------------------------------------------------===//
// TypeParamType custom print/parse
//===----------------------------------------------------------------------===//

void TypeParamType::print(AsmPrinter& printer) const {
  printer << "<\"" << getName() << "\"";
  if (getConstraint() != TypeParamKind::Any) {
    printer << ", " << stringifyTypeParamKind(getConstraint());
  }
  printer << ">";
}

Type TypeParamType::parse(AsmParser& parser) {
  std::string name;
  TypeParamKind constraint = TypeParamKind::Any;

  if (parser.parseLess()) {
    return {};
  }

  // Parse the name as a quoted string
  if (parser.parseString(&name)) {
    return {};
  }

  // Check for optional constraint
  if (succeeded(parser.parseOptionalComma())) {
    StringRef constraintStr;
    if (parser.parseKeyword(&constraintStr)) {
      return {};
    }
    auto maybeConstraint = symbolizeTypeParamKind(constraintStr);
    if (!maybeConstraint) {
      parser.emitError(parser.getCurrentLocation(),
                       "invalid type parameter constraint: ")
          << constraintStr;
      return {};
    }
    constraint = *maybeConstraint;
  }

  if (parser.parseGreater()) {
    return {};
  }

  return TypeParamType::get(parser.getContext(), name, constraint);
}

bool TypeParamType::satisfiesConstraint(mlir::Type concreteType) const {
  switch (getConstraint()) {
  case TypeParamKind::Any:
    // Any type satisfies the 'any' constraint
    return llvm::isa<IntegerType>(concreteType) ||
           llvm::isa<FloatType>(concreteType) ||
           llvm::isa<BoolType>(concreteType);
  case TypeParamKind::Numeric:
    // Only integer or float types satisfy 'numeric'
    return llvm::isa<IntegerType>(concreteType) ||
           llvm::isa<FloatType>(concreteType);
  case TypeParamKind::Integer:
    // Only integer types satisfy 'integer'
    return llvm::isa<IntegerType>(concreteType);
  case TypeParamKind::Float:
    // Only float types satisfy 'float'
    return llvm::isa<FloatType>(concreteType);
  }
  llvm_unreachable("Unknown TypeParamKind");
}

void PolangDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "polang/Dialect/PolangTypes.cpp.inc"
      >();
}
