#include "compiler/codegen.hpp"
#include "compiler/codegen_visitor.hpp"
#include "parser/node.hpp"

#include <iostream>

/* Compile the AST into a module */
void CodeGenContext::generateCode(NBlock& root) {
  /* Create the top level interpreter function to call as entry */
  std::vector<Type*> argTypes;
  // Return i64 to allow REPL to capture and print result
  FunctionType* ftype =
      FunctionType::get(Type::getInt64Ty(context), argTypes, false);
  mainFunction =
      Function::Create(ftype, GlobalValue::ExternalLinkage, "main", module);
  BasicBlock* const bblock = BasicBlock::Create(context, "entry", mainFunction, 0);

  /* Push a new variable/block context */
  pushBlock(bblock);
  CodeGenVisitor visitor(*this);
  root.accept(visitor); /* emit bytecode for the toplevel block */

  // Get the last expression value and return it as i64
  Value* const lastValue = visitor.getResult();
  Value* retVal = nullptr;

  if (lastValue != nullptr) {
    Type* const valType = lastValue->getType();

    if (valType->isDoubleTy()) {
      // Bitcast double to i64
      retVal = new BitCastInst(lastValue, Type::getInt64Ty(context), "",
                               currentBlock());
    } else if (valType->isIntegerTy(1)) {
      // Zero-extend bool (i1) to i64
      retVal = new ZExtInst(lastValue, Type::getInt64Ty(context), "",
                            currentBlock());
    } else if (valType->isIntegerTy(64)) {
      // Already i64
      retVal = lastValue;
    } else {
      // Unknown type, return 0
      retVal = ConstantInt::get(Type::getInt64Ty(context), 0);
    }
  } else {
    // No expression, return 0
    retVal = ConstantInt::get(Type::getInt64Ty(context), 0);
  }

  ReturnInst::Create(context, retVal, currentBlock());
  popBlock();
}

/* Print the LLVM IR to the given output stream */
void CodeGenContext::printIR(raw_ostream& os) const {
  module->print(os, nullptr);
}

/* Executes the AST by running the main function */
GenericValue CodeGenContext::runCode() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  // Use generic CPU to avoid SIGILL from unsupported instructions
  auto JTMB = orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB) {
    std::cerr << "Failed to detect host: " << toString(JTMB.takeError())
              << "\n";
    return GenericValue();
  }
  JTMB->setCPU("generic");

  auto JIT =
      orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(*JTMB)).create();
  if (!JIT) {
    std::cerr << "Failed to create JIT: " << toString(JIT.takeError()) << "\n";
    return GenericValue();
  }

  // Note: LLJIT takes ownership of the module
  auto TSCtx =
      std::make_unique<orc::ThreadSafeContext>(std::make_unique<LLVMContext>());
  // Clone the module into the new context for JIT
  // For simplicity, we just transfer ownership here
  auto TSM = orc::ThreadSafeModule(std::unique_ptr<Module>(module), *TSCtx);
  module = nullptr; // ownership transferred

  if (auto Err = (*JIT)->addIRModule(std::move(TSM))) {
    std::cerr << "Failed to add module: " << toString(std::move(Err)) << "\n";
    return GenericValue();
  }

  auto MainSym = (*JIT)->lookup("main");
  if (!MainSym) {
    std::cerr << "Failed to find main: " << toString(MainSym.takeError())
              << "\n";
    return GenericValue();
  }

  auto* MainFn = MainSym->toPtr<void (*)()>();
  MainFn();

  return GenericValue();
}
