//===- Passes.h - Polang transformation passes ------------------*- C++ -*-===//
//
// This file declares the transformation passes for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TRANSFORMS_PASSES_H
#define POLANG_TRANSFORMS_PASSES_H

// Suppress warnings from MLIR headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Pass/Pass.h"

#pragma GCC diagnostic pop

#include <memory>

namespace polang {

/// Create a pass to inline GlobalOp initializer regions into the entry
/// function. Must run before monomorphization.
std::unique_ptr<mlir::Pass> createInlineGlobalInitializersPass();

/// Create a pass to monomorphize polymorphic functions.
std::unique_ptr<mlir::Pass> createMonomorphizationPass();

} // namespace polang

#endif // POLANG_TRANSFORMS_PASSES_H
