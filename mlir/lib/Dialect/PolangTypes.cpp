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
    if (!llvm::isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType,
                   TypeParamType, TupleType>(elem)) {
      return emitError() << "tuple element " << i << " type (" << elem
                         << ") is not a valid Polang type or type parameter";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DataLayoutTypeInterface implementations
//===----------------------------------------------------------------------===//

::llvm::TypeSize polang::TupleType::getTypeSizeInBits(
    const ::mlir::DataLayout& dataLayout,
    ::mlir::DataLayoutEntryListRef /*params*/) const {
  auto sz = typeSize(&dataLayout);
  if (!sz) {
    return ::llvm::TypeSize::getFixed(0);
  }
  return ::llvm::TypeSize::getFixed(*sz * 8);
}

uint64_t polang::TupleType::getABIAlignment(
    const ::mlir::DataLayout& /*dataLayout*/,
    ::mlir::DataLayoutEntryListRef /*params*/) const {
  return 8;
}

uint64_t polang::TupleType::getPreferredAlignment(
    const ::mlir::DataLayout& /*dataLayout*/,
    ::mlir::DataLayoutEntryListRef /*params*/) const {
  return 8;
}

std::optional<uint64_t>
polang::getTypeSize(Type type, const ::mlir::DataLayout* dataLayout) {
  if (llvm::isa<TypeParamType>(type)) {
    return std::nullopt;
  }
  if (auto tupleType = llvm::dyn_cast<TupleType>(type)) {
    return tupleType.typeSize(dataLayout);
  }
  if (dataLayout != nullptr) {
    return dataLayout->getTypeSize(type);
  }
  mlir::DataLayout defaultDL;
  return defaultDL.getTypeSize(type);
}

std::optional<uint64_t>
polang::TupleType::typeSize(const ::mlir::DataLayout* dataLayout) const {
  uint64_t currentOffset = 0;
  for (Type elem : getTypes()) {
    currentOffset = llvm::alignTo(currentOffset, 8);
    auto sz = polang::getTypeSize(elem, dataLayout);
    if (!sz) {
      return std::nullopt;
    }
    currentOffset += *sz;
  }
  return llvm::alignTo(currentOffset, 8);
}

std::optional<uint64_t> polang::TupleType::getElementOffset(
    size_t index, const ::mlir::DataLayout* dataLayout) const {
  assert(index < getTypes().size() && "element index out of bounds");
  uint64_t currentOffset = 0;
  for (size_t i = 0; i < index; ++i) {
    currentOffset = llvm::alignTo(currentOffset, 8);
    auto sz = polang::getTypeSize(getTypes()[i], dataLayout);
    if (!sz) {
      return std::nullopt;
    }
    currentOffset += *sz;
  }
  return llvm::alignTo(currentOffset, 8);
}

std::optional<SmallVector<uint64_t>> polang::TupleType::getElementOffsets(
    const ::mlir::DataLayout* dataLayout) const {
  SmallVector<uint64_t> offsets;
  offsets.reserve(getTypes().size());
  uint64_t currentOffset = 0;
  for (Type elem : getTypes()) {
    currentOffset = llvm::alignTo(currentOffset, 8);
    offsets.push_back(currentOffset);
    auto sz = polang::getTypeSize(elem, dataLayout);
    if (!sz) {
      return std::nullopt;
    }
    currentOffset += *sz;
  }
  return offsets;
}

std::optional<SmallVector<uint64_t>> polang::TupleType::getElementSlotOffsets(
    const ::mlir::DataLayout* dataLayout) const {
  auto byteOffsets = getElementOffsets(dataLayout);
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
