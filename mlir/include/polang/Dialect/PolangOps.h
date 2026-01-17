//===- PolangOps.h - Polang operation declarations --------------*- C++ -*-===//
//
// This file declares the operations for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_DIALECT_POLANGOPS_H
#define POLANG_DIALECT_POLANGOPS_H

// Suppress Clang warnings from MLIR headers (OperationSupport.h)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdangling-assignment-gsl"
#endif

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangTypes.h"

// Include enums before ops (ops depend on enums)
#include "polang/Dialect/PolangEnums.h.inc"

#define GET_OP_CLASSES
#include "polang/Dialect/PolangOps.h.inc"

#endif // POLANG_DIALECT_POLANGOPS_H
