//===- PolangASTOps.h - Polang AST operation declarations --------*- C++ -*-===//
//
// This file declares the operations for the Polang AST dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_DIALECT_POLANGASTOPS_H
#define POLANG_DIALECT_POLANGASTOPS_H

// Suppress warnings from MLIR headers
#pragma GCC diagnostic push
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wdangling-assignment-gsl"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangASTTypes.h"
#include "polang/Dialect/PolangTypes.h"

#define GET_OP_CLASSES
#include "polang/Dialect/PolangASTOps.h.inc"

#pragma GCC diagnostic pop

#endif // POLANG_DIALECT_POLANGASTOPS_H
