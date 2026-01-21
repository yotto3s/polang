//===- PolangTypeConverter.h - Type conversion for Polang -------*- C++ -*-===//
//
// This file declares the PolangTypeConverter class for converting between
// Polang AST types and MLIR types.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_POLANGTYPECONVERTER_H
#define POLANG_POLANGTYPECONVERTER_H

#include "polang/Dialect/PolangTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

#include <cstdint>
#include <string>

class NTypeSpec;

namespace polang {

/// Converts between Polang AST types and MLIR types.
/// Provides a centralized place for all type conversion logic.
class PolangTypeConverter {
public:
  explicit PolangTypeConverter(mlir::MLIRContext* context);

  /// Generate a fresh type variable with optional kind constraint.
  [[nodiscard]] mlir::Type freshTypeVar(TypeVarKind kind = TypeVarKind::Any);

  /// Get a Polang type from annotation, or a fresh type variable if none.
  [[nodiscard]] mlir::Type getTypeOrFresh(const NTypeSpec* typeAnnotation);

  /// Get a Polang MLIR type from an NTypeSpec.
  [[nodiscard]] mlir::Type getPolangType(const NTypeSpec& typeSpec);

  /// Get the default type (i64) for cases where no type is specified.
  [[nodiscard]] mlir::Type getDefaultType();

  /// Convert a Polang MLIR type to a standard MLIR type.
  [[nodiscard]] mlir::Type convertPolangType(mlir::Type polangType);

private:
  mlir::MLIRContext* context;
  uint64_t nextTypeVarId = 0;
};

} // namespace polang

#endif // POLANG_POLANGTYPECONVERTER_H
