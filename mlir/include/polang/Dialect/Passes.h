//===- Passes.h - Polang dialect passes -------------------------*- C++ -*-===//
//
// This file declares the passes for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_DIALECT_PASSES_H
#define POLANG_DIALECT_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace polang {

/// Creates a pass to perform Polang-specific type inference.
/// This pass infers return types for functions that don't have explicit
/// return type annotations.
std::unique_ptr<mlir::Pass> createPolangTypeInferencePass();

/// Registers all Polang dialect passes.
void registerPolangDialectPasses();

} // namespace polang

#endif // POLANG_DIALECT_PASSES_H
