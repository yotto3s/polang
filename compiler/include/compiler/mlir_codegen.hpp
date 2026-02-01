//===- mlir_codegen.hpp - MLIR-based code generation ------------*- C++ -*-===//
//
// This file declares the MLIR-based code generation context.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_COMPILER_MLIR_CODEGEN_HPP
#define POLANG_COMPILER_MLIR_CODEGEN_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename T> class OwningOpRef;
} // namespace mlir

namespace llvm {
class LLVMContext;
class raw_ostream;
} // namespace llvm

class NBlock;

namespace polang {

struct CompiledSymbols;

/// Optional reference to compiled symbols for incremental compilation.
using OptCompiledSymbols =
    std::optional<std::reference_wrapper<const CompiledSymbols>>;

/// MLIR-based code generation context.
/// Generates MLIR from AST, lowers to LLVM dialect, and can execute via JIT.
class MLIRCodeGenContext {
public:
  MLIRCodeGenContext();
  ~MLIRCodeGenContext();

  /// Generate MLIR from the given AST.
  /// If emitTypeVars is true, untyped positions will emit type variables
  /// for polymorphic type inference at the MLIR level.
  /// If inferredType is non-empty, it is used as the return type of the
  /// entry function.
  /// If entryFuncName is non-empty, the entry function is named accordingly.
  /// Returns true on success.
  [[nodiscard]] bool
  generateCode(const NBlock& ast, bool emitTypeVars = false,
               const std::string& inferredType = "",
               const std::string& entryFuncName = "",
               OptCompiledSymbols compiledSymbols = std::nullopt);

  /// Run type inference pass to resolve type variables.
  /// Must be called after generateCode() when emitTypeVars was true.
  [[nodiscard]] bool runTypeInference();

  /// Lower the Polang dialect to standard dialects.
  /// Must be called after generateCode().
  [[nodiscard]] bool lowerToStandard();

  /// Lower standard dialects to LLVM dialect.
  /// Must be called after lowerToStandard().
  [[nodiscard]] bool lowerToLLVM();

  /// Print the current MLIR module.
  void printMLIR(llvm::raw_ostream& os);

  /// Convert to LLVM IR and print.
  /// Must be called after lowerToLLVM().
  [[nodiscard]] bool printLLVMIR(llvm::raw_ostream& os);

  /// Execute the code via JIT and return the result.
  /// Must be called after lowerToLLVM().
  /// Returns the result of the main function.
  [[nodiscard]] bool runCode(int64_t& result);

  /// Get the resolved return type name of an entry function.
  /// Must be called after runTypeInference().
  /// Returns "int", "double", "bool", or "unknown".
  [[nodiscard]] std::string
  getResolvedReturnType(const std::string& funcName = "__polang_entry") const;

  /// Take ownership of the MLIR module (for external JIT execution).
  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp> takeModule();

  /// Get the last error message.
  [[nodiscard]] const std::string& getError() const { return error; }

private:
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<llvm::LLVMContext> llvmContext;
  std::unique_ptr<mlir::OwningOpRef<mlir::ModuleOp>> module;
  std::string error;

  [[nodiscard]] bool initializeContext();
};

} // namespace polang

#endif // POLANG_COMPILER_MLIR_CODEGEN_HPP
