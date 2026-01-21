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

// Include the Signedness enum before the types that use it
#include "polang/Dialect/PolangEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "polang/Dialect/PolangTypes.h.inc"

#pragma GCC diagnostic pop

#endif // POLANG_DIALECT_POLANGTYPES_H
