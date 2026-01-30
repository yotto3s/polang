#ifndef POLANG_JIT_SESSION_HPP
#define POLANG_JIT_SESSION_HPP

#include <cstdint>
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

  /// Execute a function by name, returning its result as i64.
  /// For float types, pass resultType ("f32" or "f64") so the JIT uses
  /// the correct calling convention and bit-casts the result to i64.
  [[nodiscard]] bool execute(const std::string& entryName, int64_t& result,
                             std::string& error,
                             const std::string& resultType = "i64");

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace polang

#endif // POLANG_JIT_SESSION_HPP
