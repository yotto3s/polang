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
    if (!llvm::isa<IntegerType, FloatType, BoolType, IndexType, TypeParamType,
                   TupleType>(elem)) {
      return emitError() << "tuple element " << i << " type (" << elem
                         << ") is not a valid Polang type or type parameter";
    }
  }
  return success();
}

std::optional<uint64_t> polang::getTypeSize(Type type) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    return intType.typeSize();
  }
  if (auto floatType = llvm::dyn_cast<FloatType>(type)) {
    return floatType.typeSize();
  }
  if (auto boolType = llvm::dyn_cast<BoolType>(type)) {
    return boolType.typeSize();
  }
  if (auto indexType = llvm::dyn_cast<IndexType>(type)) {
    return indexType.typeSize();
  }
  if (auto tupleType = llvm::dyn_cast<TupleType>(type)) {
    return tupleType.typeSize();
  }
  return std::nullopt;
}

std::optional<uint64_t> polang::TupleType::typeSize() const {
  uint64_t currentOffset = 0;
  for (Type elem : getTypes()) {
    currentOffset = llvm::alignTo(currentOffset, 8);
    auto sz = polang::getTypeSize(elem);
    if (!sz) {
      return std::nullopt;
    }
    currentOffset += *sz;
  }
  return llvm::alignTo(currentOffset, 8);
}

std::optional<uint64_t>
polang::TupleType::getElementOffset(size_t index) const {
  assert(index < getTypes().size() && "element index out of bounds");
  uint64_t currentOffset = 0;
  for (size_t i = 0; i < index; ++i) {
    currentOffset = llvm::alignTo(currentOffset, 8);
    auto sz = polang::getTypeSize(getTypes()[i]);
    if (!sz) {
      return std::nullopt;
    }
    currentOffset += *sz;
  }
  return llvm::alignTo(currentOffset, 8);
}

std::optional<SmallVector<uint64_t>>
polang::TupleType::getElementOffsets() const {
  SmallVector<uint64_t> offsets;
  offsets.reserve(getTypes().size());
  uint64_t currentOffset = 0;
  for (Type elem : getTypes()) {
    currentOffset = llvm::alignTo(currentOffset, 8);
    offsets.push_back(currentOffset);
    auto sz = polang::getTypeSize(elem);
    if (!sz) {
      return std::nullopt;
    }
    currentOffset += *sz;
  }
  return offsets;
}

std::optional<SmallVector<uint64_t>>
polang::TupleType::getElementSlotOffsets() const {
  auto byteOffsets = getElementOffsets();
  if (!byteOffsets) {
    return std::nullopt;
  }
  SmallVector<uint64_t> slotOffsets;
  slotOffsets.reserve(byteOffsets->size());
  for (uint64_t byteOffset : *byteOffsets) {
    slotOffsets.push_back(byteOffset / 8);
  }
  return slotOffsets;
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
