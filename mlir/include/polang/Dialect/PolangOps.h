//===- PolangOps.h - Polang operation declarations --------------*- C++ -*-===//
//
// This file declares the operations for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_DIALECT_POLANGOPS_H
#define POLANG_DIALECT_POLANGOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangTypes.h"

// Include enums before ops (ops depend on enums)
#include "polang/Dialect/PolangEnums.h.inc"

#define GET_OP_CLASSES
#include "polang/Dialect/PolangOps.h.inc"

#endif // POLANG_DIALECT_POLANGOPS_H
