//===- PolangDialect.cpp - Polang dialect implementation --------*- C++ -*-===//
//
// This file implements the Polang dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers and generated code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

using namespace mlir;
using namespace polang;

#include "polang/Dialect/PolangDialect.cpp.inc"

#pragma GCC diagnostic pop

//===----------------------------------------------------------------------===//
// Polang dialect initialization
//===----------------------------------------------------------------------===//

void PolangDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "polang/Dialect/PolangOps.cpp.inc"
      >();
}
