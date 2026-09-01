//===- polang-opt.cpp - Polang MLIR optimizer driver ----------*- C++ -*-===//
//
// A minimal mlir-opt wrapper that registers Polang dialects.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "polang/Conversion/Passes.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Transforms/Passes.h"

#pragma GCC diagnostic pop

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<polang::PolangDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::cf::ControlFlowDialect>();

  polang::registerPolangTransformsPasses();
  polang::registerPolangConversionPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Polang optimizer", registry));
}
