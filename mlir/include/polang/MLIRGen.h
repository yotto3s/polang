//===- MLIRGen.h - MLIR Generation from Polang AST --------------*- C++ -*-===//
//
// This file declares the interface for generating MLIR from the Polang AST.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_MLIRGEN_H
#define POLANG_MLIRGEN_H

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename OpTy> class OwningOpRef;
} // namespace mlir

class NBlock;

namespace polang {

struct CompiledSymbols;

/// Optional reference to compiled symbols for incremental compilation.
using OptCompiledSymbols =
    std::optional<std::reference_wrapper<const CompiledSymbols>>;

/// Generate MLIR from a Polang AST.
/// Returns nullptr on failure.
/// If emitTypeVars is true, untyped positions will emit type variables
/// for polymorphic type inference at the MLIR level.
/// If inferredType is non-empty, it is used as the return type of the
/// entry function.
/// If entryFuncName is non-empty, the entry function is named accordingly
/// instead of the default "__polang_entry".
/// If compiledSymbols is provided, incremental mode is used: extern
/// declarations are emitted for previously compiled symbols, and only
/// new statements generate MLIR code.
mlir::OwningOpRef<mlir::ModuleOp>
mlirGen(mlir::MLIRContext& context, const NBlock& moduleAST,
        bool emitTypeVars = false, const std::string& inferredType = "",
        const std::string& entryFuncName = "",
        OptCompiledSymbols compiledSymbols = std::nullopt);

} // namespace polang

#endif // POLANG_MLIRGEN_H
