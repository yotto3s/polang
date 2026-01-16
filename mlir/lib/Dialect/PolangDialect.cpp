//===- PolangDialect.cpp - Polang dialect implementation --------*- C++ -*-===//
//
// This file implements the Polang dialect.
//
//===----------------------------------------------------------------------===//

#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

using namespace mlir;
using namespace polang;

#include "polang/Dialect/PolangDialect.cpp.inc"

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
