#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRPrinter/IRPrintingPasses.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <stack>

using namespace llvm;

class NBlock;

class CodeGenBlock {
public:
  BasicBlock *block;
  std::map<std::string, AllocaInst *> locals;
};

class CodeGenContext {
  std::stack<CodeGenBlock *> blocks;
  Function *mainFunction;

public:
  LLVMContext context;
  Module *module;
  CodeGenContext() { module = new Module("main", context); }

  void generateCode(NBlock &root);
  void printIR(raw_ostream &os);
  GenericValue runCode();
  std::map<std::string, AllocaInst *> &locals() { return blocks.top()->locals; }
  BasicBlock *currentBlock() { return blocks.top()->block; }
  void pushBlock(BasicBlock *block) {
    blocks.push(new CodeGenBlock());
    blocks.top()->block = block;
  }
  void popBlock() {
    CodeGenBlock *top = blocks.top();
    blocks.pop();
    delete top;
  }
};
