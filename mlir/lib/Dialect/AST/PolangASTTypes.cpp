//===- PolangASTTypes.cpp - Polang AST type implementation -------*- C++ -*-===//
//
// This file implements the types for the Polang AST dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangASTTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "polang/Dialect/PolangASTDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace polang::ast;

#define GET_TYPEDEF_CLASSES
#include "polang/Dialect/PolangASTTypes.cpp.inc"

#pragma GCC diagnostic pop

//===----------------------------------------------------------------------===//
// TypeVarType custom print/parse
//===----------------------------------------------------------------------===//

void TypeVarType::print(AsmPrinter& printer) const {
  printer << "<" << getId();
  if (getKind() != ::polang::TypeVarKind::Any) {
    printer << ", " << ::polang::stringifyTypeVarKind(getKind());
  }
  printer << ">";
}

Type TypeVarType::parse(AsmParser& parser) {
  uint64_t id = 0;
  ::polang::TypeVarKind kind = ::polang::TypeVarKind::Any;

  if (parser.parseLess() || parser.parseInteger(id)) {
    return {};
  }

  // Check for optional kind
  if (succeeded(parser.parseOptionalComma())) {
    StringRef kindStr;
    if (parser.parseKeyword(&kindStr)) {
      return {};
    }
    auto maybeKind = ::polang::symbolizeTypeVarKind(kindStr);
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

void PolangASTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "polang/Dialect/PolangASTTypes.cpp.inc"
      >();
}
