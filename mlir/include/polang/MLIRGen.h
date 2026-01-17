//===- MLIRGen.h - MLIR Generation from Polang AST --------------*- C++ -*-===//
//
// This file declares the interface for generating MLIR from the Polang AST.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_MLIRGEN_H
#define POLANG_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename OpTy>
class OwningOpRef;
} // namespace mlir

class NBlock;

namespace polang {

/// Generate MLIR from a Polang AST.
/// Returns nullptr on failure.
/// If emitTypeVars is true, untyped positions will emit type variables
/// for polymorphic type inference at the MLIR level.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                           const NBlock &moduleAST,
                                           bool emitTypeVars = false);

} // namespace polang

#endif // POLANG_MLIRGEN_H
