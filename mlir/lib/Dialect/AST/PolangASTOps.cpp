//===- PolangASTOps.cpp - Polang AST operation implementation ----*- C++ -*-===//
//
// This file implements the operations for the Polang AST dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers and generated code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangASTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangASTTypes.h"
#include "polang/Dialect/PolangTypes.h"

using namespace mlir;
using namespace polang::ast;

#define GET_OP_CLASSES
#include "polang/Dialect/PolangASTOps.cpp.inc"

#pragma GCC diagnostic pop

//===----------------------------------------------------------------------===//
// Operation implementations will be added in subsequent steps
//===----------------------------------------------------------------------===//
