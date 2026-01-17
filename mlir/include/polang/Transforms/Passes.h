//===- Passes.h - Polang transformation passes ------------------*- C++ -*-===//
//
// This file declares the transformation passes for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TRANSFORMS_PASSES_H
#define POLANG_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace polang {

/// Create a pass to infer types for type variables using Hindley-Milner.
std::unique_ptr<mlir::Pass> createTypeInferencePass();

/// Create a pass to monomorphize polymorphic functions.
std::unique_ptr<mlir::Pass> createMonomorphizationPass();

} // namespace polang

#endif // POLANG_TRANSFORMS_PASSES_H
