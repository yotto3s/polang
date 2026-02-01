//===- jit_session.cpp - Persistent JIT session -----------------*- C++ -*-===//
//
// This file implements the persistent JIT session for the REPL.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "compiler/jit_session.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#pragma GCC diagnostic pop

#include <mutex>

using namespace mlir;

namespace {

/// Ensure LLVM targets are initialized exactly once.
void ensureLLVMTargetsInitialized() {
  static std::once_flag llvmInitFlag;
  std::call_once(llvmInitFlag, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
}

} // namespace

namespace polang {

struct JITSession::Impl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  int moduleCounter = 0;
  // Track the most recent JITDylib for symbol lookup
  llvm::orc::JITDylib* latestDylib = nullptr;
  // Track all previous dylibs for link order
  std::vector<llvm::orc::JITDylib*> allDylibs;
};

JITSession::JITSession() : impl(std::make_unique<Impl>()) {}
JITSession::~JITSession() = default;

bool JITSession::initialize(std::string& error) {
  ensureLLVMTargetsInitialized();

  auto jitBuilder = llvm::orc::LLJITBuilder();
  auto jitExpected = jitBuilder.create();
  if (!jitExpected) {
    error =
        "Failed to create LLJIT: " + llvm::toString(jitExpected.takeError());
    return false;
  }

  impl->jit = std::move(*jitExpected);
  impl->latestDylib = &impl->jit->getMainJITDylib();
  impl->allDylibs.push_back(impl->latestDylib);
  return true;
}

bool JITSession::addModule(mlir::OwningOpRef<mlir::ModuleOp>& module,
                           std::string& error) {
  if (!impl->jit) {
    error = "JIT not initialized";
    return false;
  }

  // Create a new LLVM context for this module
  auto llvmCtx = std::make_unique<llvm::LLVMContext>();

  // Translate MLIR module to LLVM IR
  auto llvmModule = translateModuleToLLVMIR(*module, *llvmCtx);
  if (!llvmModule) {
    error = "Failed to translate MLIR to LLVM IR";
    return false;
  }

  // Set data layout from JIT
  llvmModule->setDataLayout(impl->jit->getDataLayout());

  // Create a ThreadSafeModule
  auto tsm =
      llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmCtx));

  // Create a new JITDylib for each module to avoid duplicate symbol errors.
  // Each new dylib links against the previous one, so symbols from earlier
  // evaluations are visible.
  const std::string dylibName =
      "polang_eval_" + std::to_string(impl->moduleCounter++);
  auto& es = impl->jit->getExecutionSession();
  auto expectedDylib = es.createJITDylib(dylibName);
  if (!expectedDylib) {
    error = "Failed to create JITDylib: " +
            llvm::toString(expectedDylib.takeError());
    return false;
  }

  auto& newDylib = *expectedDylib;

  // Link the new dylib against ALL previous dylibs so it can resolve
  // symbols from any prior evaluation (link order is not transitive).
  for (auto* dylib : impl->allDylibs) {
    newDylib.addToLinkOrder(*dylib);
  }

  if (auto err = impl->jit->addIRModule(newDylib, std::move(tsm))) {
    error = "Failed to add module to JIT: " + llvm::toString(std::move(err));
    return false;
  }

  impl->allDylibs.push_back(&newDylib);
  impl->latestDylib = &newDylib;
  return true;
}

std::optional<JitResult> JITSession::execute(const std::string& entryName,
                                             std::string& error,
                                             const std::string& resultType) {
  if (!impl->jit) {
    error = "JIT not initialized";
    return std::monostate{};
  }

  // Look up the entry function in the latest dylib
  auto sym = impl->jit->lookup(*impl->latestDylib, entryName);
  if (!sym) {
    error = "Failed to find entry function '" + entryName +
            "': " + llvm::toString(sym.takeError());
    return std::monostate{};
  }

  // Use the correct function pointer type based on the return type.
  // Float/double returns use SSE registers (xmm0), while integer returns
  // use the general-purpose register (rax). Narrow integers must be called
  // with the correct return type to avoid garbage in upper bits.
  if (resultType == "f64") {
    auto entryFn = sym->toPtr<double (*)()>();
    return entryFn();
  }
  if (resultType == "f32") {
    auto entryFn = sym->toPtr<float (*)()>();
    return entryFn();
  }
  if (resultType == "bool") {
    auto entryFn = sym->toPtr<int8_t (*)()>();
    return static_cast<bool>(entryFn());
  }
  if (resultType == "i8" || resultType == "u8") {
    auto entryFn = sym->toPtr<int8_t (*)()>();
    return entryFn();
  }
  if (resultType == "i16" || resultType == "u16") {
    auto entryFn = sym->toPtr<int16_t (*)()>();
    return entryFn();
  }
  if (resultType == "i32" || resultType == "u32") {
    auto entryFn = sym->toPtr<int32_t (*)()>();
    return static_cast<int64_t>(entryFn());
  }
  auto entryFn = sym->toPtr<int64_t (*)()>();
  return entryFn();
}

} // namespace polang
