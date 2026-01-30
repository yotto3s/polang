#ifndef POLANG_DIALECT_POLANGLOCATIONS_H
#define POLANG_DIALECT_POLANGLOCATIONS_H

#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"

// Forward declaration from parser
struct SourceLocation;

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace polang {

/// Create a FileLineColLoc from source location.
/// Returns UnknownLoc if srcLoc is invalid.
[[nodiscard]] mlir::Location getFileLoc(mlir::MLIRContext* ctx,
                                        llvm::StringRef filename,
                                        const SourceLocation& srcLoc);

/// Create a NameLoc wrapping a base location (for variables, functions).
/// Used to associate a name with a source location.
[[nodiscard]] mlir::Location getNameLoc(mlir::MLIRContext* ctx,
                                        llvm::StringRef name,
                                        mlir::Location baseLoc);

/// Create a CallSiteLoc for function calls.
/// Associates the callee location with the caller location.
[[nodiscard]] mlir::Location getCallSiteLoc(mlir::Location calleeLoc,
                                            mlir::Location callerLoc);

/// Get unknown location.
[[nodiscard]] mlir::Location getUnknownLoc(mlir::MLIRContext* ctx);

} // namespace polang

#endif // POLANG_DIALECT_POLANGLOCATIONS_H
