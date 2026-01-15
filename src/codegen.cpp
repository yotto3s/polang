#include "codegen.hpp"
#include "node.hpp"
#include "parser.hpp"

/* Compile the AST into a module */
void CodeGenContext::generateCode(NBlock &root) {
  std::cout << "Generating code...\n";

  /* Create the top level interpreter function to call as entry */
  std::vector<Type *> argTypes;
  FunctionType *ftype =
      FunctionType::get(Type::getVoidTy(context), argTypes, false);
  mainFunction =
      Function::Create(ftype, GlobalValue::ExternalLinkage, "main", module);
  BasicBlock *bblock = BasicBlock::Create(context, "entry", mainFunction, 0);

  /* Push a new variable/block context */
  pushBlock(bblock);
  root.codeGen(*this); /* emit bytecode for the toplevel block */
  ReturnInst::Create(context, bblock);
  popBlock();

  /* Print the bytecode in a human-readable format
     to see if our program compiled properly
   */
  std::cout << "Code is generated.\n";
  module->print(outs(), nullptr);
}

/* Executes the AST by running the main function */
GenericValue CodeGenContext::runCode() {
  std::cout << "Running code...\n";

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

  auto *MainFn = MainSym->toPtr<void (*)()>();
  MainFn();

  std::cout << "Code was run.\n";
  return GenericValue();
}

/* Returns an LLVM type based on the identifier */
static Type *typeOf(const NIdentifier &type, LLVMContext &ctx) {
  if (type.name.compare("int") == 0) {
    return Type::getInt64Ty(ctx);
  } else if (type.name.compare("double") == 0) {
    return Type::getDoubleTy(ctx);
  }
  return Type::getVoidTy(ctx);
}

/* -- Code Generation -- */

Value *NInteger::codeGen(CodeGenContext &context) {
  std::cout << "Creating integer: " << value << std::endl;
  return ConstantInt::get(Type::getInt64Ty(context.context), value, true);
}

Value *NDouble::codeGen(CodeGenContext &context) {
  std::cout << "Creating double: " << value << std::endl;
  return ConstantFP::get(Type::getDoubleTy(context.context), value);
}

Value *NIdentifier::codeGen(CodeGenContext &context) {
  std::cout << "Creating identifier reference: " << name << std::endl;
  if (context.locals().find(name) == context.locals().end()) {
    std::cerr << "undeclared variable " << name << std::endl;
    return NULL;
  }
  return new LoadInst(context.locals()[name]->getAllocatedType(),
                      context.locals()[name], "", false,
                      context.currentBlock());
}

Value *NMethodCall::codeGen(CodeGenContext &context) {
  Function *function = context.module->getFunction(id.name.c_str());
  if (function == NULL) {
    std::cerr << "no such function " << id.name << std::endl;
  }
  std::vector<Value *> args;
  ExpressionList::const_iterator it;
  for (it = arguments.begin(); it != arguments.end(); it++) {
    args.push_back((**it).codeGen(context));
  }
  CallInst *call = CallInst::Create(function->getFunctionType(), function, args,
                                    "", context.currentBlock());
  std::cout << "Creating method call: " << id.name << std::endl;
  return call;
}

Value *NBinaryOperator::codeGen(CodeGenContext &context) {
  std::cout << "Creating binary operation " << op << std::endl;
  Instruction::BinaryOps instr;
  switch (op) {
  case TPLUS:
    instr = Instruction::Add;
    goto math;
  case TMINUS:
    instr = Instruction::Sub;
    goto math;
  case TMUL:
    instr = Instruction::Mul;
    goto math;
  case TDIV:
    instr = Instruction::SDiv;
    goto math;

    /* TODO comparison */
  }

  return NULL;
math:
  return BinaryOperator::Create(instr, lhs.codeGen(context),
                                rhs.codeGen(context), "",
                                context.currentBlock());
}

Value *NAssignment::codeGen(CodeGenContext &context) {
  std::cout << "Creating assignment for " << lhs.name << std::endl;
  if (context.locals().find(lhs.name) == context.locals().end()) {
    std::cerr << "undeclared variable " << lhs.name << std::endl;
    return NULL;
  }
  return new StoreInst(rhs.codeGen(context), context.locals()[lhs.name], false,
                       context.currentBlock());
}

Value *NBlock::codeGen(CodeGenContext &context) {
  StatementList::const_iterator it;
  Value *last = NULL;
  for (it = statements.begin(); it != statements.end(); it++) {
    std::cout << "Generating code for " << typeid(**it).name() << std::endl;
    last = (**it).codeGen(context);
  }
  std::cout << "Creating block" << std::endl;
  return last;
}

Value *NExpressionStatement::codeGen(CodeGenContext &context) {
  std::cout << "Generating code for " << typeid(expression).name() << std::endl;
  return expression.codeGen(context);
}

Value *NVariableDeclaration::codeGen(CodeGenContext &context) {
  std::cout << "Creating variable declaration " << type.name << " " << id.name
            << std::endl;
  AllocaInst *alloc = new AllocaInst(typeOf(type, context.context), 0,
                                     id.name.c_str(), context.currentBlock());
  context.locals()[id.name] = alloc;
  if (assignmentExpr != NULL) {
    NAssignment assn(id, *assignmentExpr);
    assn.codeGen(context);
  }
  return alloc;
}

Value *NFunctionDeclaration::codeGen(CodeGenContext &context) {
  std::vector<Type *> argTypes;
  VariableList::const_iterator it;
  for (it = arguments.begin(); it != arguments.end(); it++) {
    argTypes.push_back(typeOf((**it).type, context.context));
  }
  FunctionType *ftype =
      FunctionType::get(typeOf(type, context.context), argTypes, false);
  Function *function = Function::Create(ftype, GlobalValue::InternalLinkage,
                                        id.name.c_str(), context.module);
  BasicBlock *bblock =
      BasicBlock::Create(context.context, "entry", function, 0);

  context.pushBlock(bblock);

  for (it = arguments.begin(); it != arguments.end(); it++) {
    (**it).codeGen(context);
  }

  block.codeGen(context);
  ReturnInst::Create(context.context, bblock);

  context.popBlock();
  std::cout << "Creating function: " << id.name << std::endl;
  return function;
}
