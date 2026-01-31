#ifndef POLANG_JIT_SESSION_HPP
#define POLANG_JIT_SESSION_HPP

#include "compiler/jit_result.hpp"

#include <memory>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename T> class OwningOpRef;
} // namespace mlir

namespace polang {

/// Persistent JIT session that supports incremental module addition.
/// Each evaluation adds a new LLVM module; symbols from previous modules
/// are automatically visible to later ones.
class JITSession {
public:
  JITSession();
  ~JITSession();

  /// Initialize the JIT. Must be called before addModule/execute.
  [[nodiscard]] bool initialize(std::string& error);

  /// Lower an MLIR module to LLVM IR and add it to the JIT.
  /// The MLIR module must already be lowered to LLVM dialect.
  [[nodiscard]] bool addModule(mlir::OwningOpRef<mlir::ModuleOp>& module,
                               std::string& error);

  /// Execute a function by name, returning its result as a type-safe variant.
  /// The resultType string determines which concrete type to extract from the
  /// JIT return value (e.g., "f64", "f32", "bool", "i8", "i16", "i64").
  [[nodiscard]] bool execute(const std::string& entryName, JitResult& result,
                             std::string& error,
                             const std::string& resultType = "i64");

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace polang

#endif // POLANG_JIT_SESSION_HPP
