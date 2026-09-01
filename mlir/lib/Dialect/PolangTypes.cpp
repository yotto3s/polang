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

namespace {

struct BuiltinTupleDataLayoutModel
    : public DataLayoutTypeInterface::ExternalModel<BuiltinTupleDataLayoutModel,
                                                    mlir::TupleType> {
  [[nodiscard]] llvm::TypeSize
  getTypeSize(mlir::Type type, const ::mlir::DataLayout& dataLayout,
              ::mlir::DataLayoutEntryListRef /*params*/) const {
    auto tupleType = llvm::cast<mlir::TupleType>(type);
    auto sz = polang::getTupleTypeSize(tupleType, &dataLayout);
    if (!sz) {
      return llvm::TypeSize::getFixed(0);
    }
    return llvm::TypeSize::getFixed(*sz);
  }

  [[nodiscard]] llvm::TypeSize
  getTypeSizeInBits(mlir::Type type, const ::mlir::DataLayout& dataLayout,
                    ::mlir::DataLayoutEntryListRef params) const {
    return getTypeSize(type, dataLayout, params) * 8;
  }

  [[nodiscard]] uint64_t
  getABIAlignment(mlir::Type /*type*/, const ::mlir::DataLayout& /*dataLayout*/,
                  ::mlir::DataLayoutEntryListRef /*params*/) const {
    return 8;
  }

  [[nodiscard]] uint64_t
  getPreferredAlignment(mlir::Type /*type*/,
                        const ::mlir::DataLayout& /*dataLayout*/,
                        ::mlir::DataLayoutEntryListRef /*params*/) const {
    return 8;
  }
};

} // namespace

void polang::registerTupleDataLayoutInterface(MLIRContext* context) {
  mlir::TupleType::attachInterface<BuiltinTupleDataLayoutModel>(*context);
}

std::optional<uint64_t>
polang::getTupleTypeSize(mlir::TupleType tupleType,
                         const ::mlir::DataLayout* dataLayout) {
  uint64_t currentOffset = 0;
  for (Type elem : tupleType.getTypes()) {
    currentOffset = llvm::alignTo(currentOffset, 8);
    auto sz = polang::getTypeSize(elem, dataLayout);
    if (!sz) {
      return std::nullopt;
    }
    currentOffset += *sz;
  }
  return llvm::alignTo(currentOffset, 8);
}

std::optional<uint64_t>
polang::getTupleElementOffset(mlir::TupleType tupleType, size_t index,
                              const ::mlir::DataLayout* dataLayout) {
  assert(index < tupleType.getTypes().size() && "element index out of bounds");
  uint64_t currentOffset = 0;
  for (size_t i = 0; i < index; ++i) {
    currentOffset = llvm::alignTo(currentOffset, 8);
    auto sz = polang::getTypeSize(tupleType.getType(i), dataLayout);
    if (!sz) {
      return std::nullopt;
    }
    currentOffset += *sz;
  }
  return llvm::alignTo(currentOffset, 8);
}

std::optional<uint64_t>
polang::getTupleElementSlotOffset(mlir::TupleType tupleType, size_t index,
                                  const ::mlir::DataLayout* dataLayout) {
  auto off = getTupleElementOffset(tupleType, index, dataLayout);
  if (!off) {
    return std::nullopt;
  }
  return *off / 8;
}

std::optional<SmallVector<uint64_t>>
polang::getTupleElementOffsets(mlir::TupleType tupleType,
                               const ::mlir::DataLayout* dataLayout) {
  SmallVector<uint64_t> offsets;
  offsets.reserve(tupleType.getTypes().size());
  uint64_t currentOffset = 0;
  for (Type elem : tupleType.getTypes()) {
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

std::optional<SmallVector<uint64_t>>
polang::getTupleElementSlotOffsets(mlir::TupleType tupleType,
                                   const ::mlir::DataLayout* dataLayout) {
  auto byteOffsets = getTupleElementOffsets(tupleType, dataLayout);
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

std::optional<uint64_t>
polang::getTypeSize(Type type, const ::mlir::DataLayout* dataLayout) {
  if (llvm::isa<TypeParamType>(type)) {
    return std::nullopt;
  }
  if (auto tupleType = llvm::dyn_cast<mlir::TupleType>(type)) {
    return getTupleTypeSize(tupleType, dataLayout);
  }
  if (dataLayout != nullptr) {
    return dataLayout->getTypeSize(type);
  }
  mlir::DataLayout defaultDL;
  return defaultDL.getTypeSize(type);
}

void PolangDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "polang/Dialect/PolangTypes.cpp.inc"
      >();
}
