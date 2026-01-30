//===- MLIRGen.h - MLIR Generation from Polang AST --------------*- C++ -*-===//
//
// This file declares the interface for generating MLIR from the Polang AST.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_MLIRGEN_H
#define POLANG_MLIRGEN_H

#include <memory>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename OpTy> class OwningOpRef;
} // namespace mlir

class NBlock;

namespace polang {

/// Generate MLIR from a Polang AST.
/// Returns nullptr on failure.
/// If emitTypeVars is true, untyped positions will emit type variables
/// for polymorphic type inference at the MLIR level.
/// If skipTypeCheck is true, type checking is skipped (assumes AST nodes
/// already have their types set by an external TypeChecker).
/// If inferredType is non-empty, it is used as the return type of the
/// entry function instead of running the internal TypeChecker.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext& context,
                                          const NBlock& moduleAST,
                                          bool emitTypeVars = false,
                                          bool skipTypeCheck = false,
                                          const std::string& inferredType = "");

} // namespace polang

#endif // POLANG_MLIRGEN_H
