//===- Passes.h - Polang conversion passes ----------------------*- C++ -*-===//
//
// This file declares the conversion passes for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_CONVERSION_PASSES_H
#define POLANG_CONVERSION_PASSES_H

// Suppress warnings from MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Pass/Pass.h"

#pragma GCC diagnostic pop

#include <memory>

namespace polang {

/// Creates a pass to convert AST dialect to Polang dialect.
std::unique_ptr<mlir::Pass> createASTToPolangPass();

/// Creates a pass to lower Polang dialect to standard dialects.
std::unique_ptr<mlir::Pass> createPolangToStandardPass();

/// Registers all Polang conversion passes.
void registerPolangConversionPasses();

} // namespace polang

#endif // POLANG_CONVERSION_PASSES_H
