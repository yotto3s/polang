//===- PolangLocations.cpp - Location helpers for Polang MLIR --*- C++ -*-===//
//
// This file implements location helper utilities for the polang_ast dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangLocations.h"
#include "mlir/IR/BuiltinAttributes.h"

#pragma GCC diagnostic pop

#include "parser/node.hpp"

namespace polang {

mlir::Location getFileLoc(mlir::MLIRContext* ctx, llvm::StringRef filename,
                          const SourceLocation& srcLoc) {
  if (srcLoc.isValid()) {
    return mlir::FileLineColLoc::get(ctx, filename, srcLoc.line, srcLoc.column);
  }
  return mlir::UnknownLoc::get(ctx);
}

mlir::Location getNameLoc(mlir::MLIRContext* ctx, llvm::StringRef name,
                          mlir::Location baseLoc) {
  return mlir::NameLoc::get(mlir::StringAttr::get(ctx, name), baseLoc);
}

mlir::Location getCallSiteLoc(mlir::Location calleeLoc,
                              mlir::Location callerLoc) {
  return mlir::CallSiteLoc::get(calleeLoc, callerLoc);
}

mlir::Location getUnknownLoc(mlir::MLIRContext* ctx) {
  return mlir::UnknownLoc::get(ctx);
}

} // namespace polang
