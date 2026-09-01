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

LogicalResult
polang::TupleType::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<Type> types) {
  for (size_t i = 0; i < types.size(); ++i) {
    Type elem = types[i];
    if (auto intType = llvm::dyn_cast<polang::IntegerType>(elem)) {
      if (intType.getWidth() > 64) {
        return emitError() << "tuple element " << i
                           << " integer width must be <= 64, but got "
                           << intType.getWidth();
      }
      continue;
    }
    if (auto floatType = llvm::dyn_cast<polang::FloatType>(elem)) {
      if (floatType.getWidth() > 64) {
        return emitError() << "tuple element " << i
                           << " float width must be <= 64, but got "
                           << floatType.getWidth();
      }
      continue;
    }
    if (llvm::isa<BoolType, polang::IndexType, TypeParamType>(elem)) {
      continue;
    }
    return emitError() << "tuple element " << i << " type (" << elem
                       << ") must be a primitive type or type parameter";
  }
  return success();
}

Type polang::TupleType::parse(AsmParser& parser) {
  if (parser.parseLess()) {
    return {};
  }
  if (succeeded(parser.parseOptionalGreater())) {
    return get(parser.getContext(), ArrayRef<Type>{});
  }
  SmallVector<Type> elementTypes;
  if (parser.parseTypeList(elementTypes) || parser.parseGreater()) {
    return {};
  }
  return getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), elementTypes);
}

void polang::TupleType::print(AsmPrinter& printer) const {
  printer << "<";
  llvm::interleaveComma(getTypes(), printer);
  printer << ">";
}

void PolangDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "polang/Dialect/PolangTypes.cpp.inc"
      >();
}
