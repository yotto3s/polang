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
#include "mlir/IR/Types.h"

#include <optional>

// Include the Signedness enum before the types that use it
#include "polang/Dialect/PolangEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "polang/Dialect/PolangTypes.h.inc"

#pragma GCC diagnostic pop

namespace polang {

/// Returns the size in bytes of the given Polang type, or std::nullopt if the
/// type is not a sized Polang type (e.g. unspecialized TypeParamType).
std::optional<uint64_t> getTypeSize(mlir::Type type);

} // namespace polang

#endif // POLANG_DIALECT_POLANGTYPES_H
