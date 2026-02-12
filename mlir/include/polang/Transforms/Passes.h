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

/// Create a pass to monomorphize polymorphic functions.
std::unique_ptr<mlir::Pass> createMonomorphizationPass();

/// Create a pass to insert overflow checks for integer arithmetic.
std::unique_ptr<mlir::Pass> createInsertOverflowChecksPass();

} // namespace polang

#endif // POLANG_TRANSFORMS_PASSES_H
