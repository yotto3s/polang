//===- PolangASTDialect.cpp - Polang AST dialect implementation --*- C++ -*-===//
//
// This file implements the Polang AST dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers and generated code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangASTOps.h"
#include "polang/Dialect/PolangASTTypes.h"

using namespace mlir;
using namespace polang::ast;

#include "polang/Dialect/PolangASTDialect.cpp.inc"

#pragma GCC diagnostic pop

//===----------------------------------------------------------------------===//
// Polang AST dialect initialization
//===----------------------------------------------------------------------===//

void PolangASTDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "polang/Dialect/PolangASTOps.cpp.inc"
      >();
}
