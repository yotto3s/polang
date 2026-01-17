//===- mlir_codegen.hpp - MLIR-based code generation ------------*- C++ -*-===//
//
// This file declares the MLIR-based code generation context.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_COMPILER_MLIR_CODEGEN_HPP
#define POLANG_COMPILER_MLIR_CODEGEN_HPP

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename T>
class OwningOpRef;
} // namespace mlir

namespace llvm {
class LLVMContext;
class raw_ostream;
}

class NBlock;

namespace polang {

/// MLIR-based code generation context.
/// Generates MLIR from AST, lowers to LLVM dialect, and can execute via JIT.
class MLIRCodeGenContext {
public:
  MLIRCodeGenContext();
  ~MLIRCodeGenContext();

  /// Generate MLIR from the given AST.
  /// If emitTypeVars is true, untyped positions will emit type variables
  /// for polymorphic type inference at the MLIR level.
  /// Returns true on success.
  bool generateCode(const NBlock &ast, bool emitTypeVars = false);

  /// Run type inference pass to resolve type variables.
  /// Must be called after generateCode() when emitTypeVars was true.
  bool runTypeInference();

  /// Lower the Polang dialect to standard dialects.
  /// Must be called after generateCode().
  bool lowerToStandard();

  /// Lower standard dialects to LLVM dialect.
  /// Must be called after lowerToStandard().
  bool lowerToLLVM();

  /// Print the current MLIR module.
  void printMLIR(llvm::raw_ostream &os);

  /// Convert to LLVM IR and print.
  /// Must be called after lowerToLLVM().
  bool printLLVMIR(llvm::raw_ostream &os);

  /// Execute the code via JIT and return the result.
  /// Must be called after lowerToLLVM().
  /// Returns the result of the main function.
  bool runCode(int64_t &result);

  /// Get the resolved return type name of the entry function.
  /// Must be called after runTypeInference().
  /// Returns "int", "double", "bool", or "unknown".
  std::string getResolvedReturnType() const;

  /// Get the last error message.
  const std::string &getError() const { return error_; }

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<llvm::LLVMContext> llvmContext_;
  std::unique_ptr<mlir::OwningOpRef<mlir::ModuleOp>> module_;
  std::string error_;

  bool initializeContext();
};

} // namespace polang

#endif // POLANG_COMPILER_MLIR_CODEGEN_HPP
