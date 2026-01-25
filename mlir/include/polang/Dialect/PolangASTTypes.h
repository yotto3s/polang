//===- PolangASTTypes.h - Polang AST type declarations -----------*- C++ -*-===//
//
// This file declares the types for the Polang AST dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_DIALECT_POLANGASTTYPES_H
#define POLANG_DIALECT_POLANGASTTYPES_H

// Suppress warnings from MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

// Include PolangTypes to get TypeVarKind enum (avoids double inclusion of PolangEnums.h.inc)
#include "polang/Dialect/PolangTypes.h"

#define GET_TYPEDEF_CLASSES
#include "polang/Dialect/PolangASTTypes.h.inc"

#pragma GCC diagnostic pop

#endif // POLANG_DIALECT_POLANGASTTYPES_H
