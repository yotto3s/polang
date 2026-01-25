//===- polang-opt.cpp - Polang MLIR optimizer driver ----------*- C++ -*-===//
//
// A minimal mlir-opt wrapper that registers Polang dialects.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangDialect.h"

#pragma GCC diagnostic pop

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<polang::PolangDialect>();
  registry.insert<polang::ast::PolangASTDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Polang optimizer", registry));
}
