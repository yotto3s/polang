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
// RefType custom print/parse
//===----------------------------------------------------------------------===//

void RefType::print(AsmPrinter& printer) const {
  printer << "<" << getElementType();
  if (getIsMutable()) {
    printer << ", mutable";
  }
  printer << ">";
}

Type RefType::parse(AsmParser& parser) {
  Type elementType;
  bool isMutable = false;

  if (parser.parseLess() || parser.parseType(elementType)) {
    return {};
  }

  // Check for optional 'mutable' keyword
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("mutable")) {
      return {};
    }
    isMutable = true;
  }

  if (parser.parseGreater()) {
    return {};
  }

  return RefType::get(parser.getContext(), elementType, isMutable);
}

void PolangDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "polang/Dialect/PolangTypes.cpp.inc"
      >();
}
