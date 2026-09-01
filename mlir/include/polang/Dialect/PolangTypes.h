//===- PolangTypes.h - Polang type declarations -----------------*- C++ -*-===//
//
// This file declares the types for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_DIALECT_POLANGTYPES_H
#define POLANG_DIALECT_POLANGTYPES_H

// Suppress warnings from MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include <optional>

#define GET_TYPEDEF_CLASSES
#include "polang/Dialect/PolangTypes.h.inc"

#pragma GCC diagnostic pop

namespace polang {

/// Returns the size in bytes of the given type, or std::nullopt if the
/// type contains unspecialized type parameters.
std::optional<uint64_t>
getTypeSize(mlir::Type type, const mlir::DataLayout *dataLayout = nullptr);

/// Returns the size in bytes of a tuple type (computed with 64-bit member
/// alignment), or std::nullopt if unspecialized.
std::optional<uint64_t>
getTupleTypeSize(mlir::TupleType tupleType, const mlir::DataLayout *dataLayout = nullptr);

/// Returns the offset in bytes of the element at the specified index within a tuple,
/// or std::nullopt if unspecialized.
std::optional<uint64_t>
getTupleElementOffset(mlir::TupleType tupleType, size_t index, const mlir::DataLayout *dataLayout = nullptr);

/// Returns the offset in 64-bit slots of the element at the specified index within a tuple,
/// or std::nullopt if unspecialized.
std::optional<uint64_t>
getTupleElementSlotOffset(mlir::TupleType tupleType, size_t index, const mlir::DataLayout *dataLayout = nullptr);

/// Returns the offsets in bytes of all elements within a tuple, or std::nullopt if unspecialized.
std::optional<llvm::SmallVector<uint64_t>>
getTupleElementOffsets(mlir::TupleType tupleType, const mlir::DataLayout *dataLayout = nullptr);

/// Returns the offsets in 64-bit slots of all elements within a tuple, or std::nullopt if unspecialized.
std::optional<llvm::SmallVector<uint64_t>>
getTupleElementSlotOffsets(mlir::TupleType tupleType, const mlir::DataLayout *dataLayout = nullptr);

/// Register DataLayoutTypeInterface external model on builtin TupleType.
void registerTupleDataLayoutInterface(mlir::MLIRContext *context);

} // namespace polang

#endif // POLANG_DIALECT_POLANGTYPES_H
