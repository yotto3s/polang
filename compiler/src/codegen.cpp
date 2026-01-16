#include "compiler/codegen.hpp"
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison union types

/* Compile the AST into a module */
void CodeGenContext::generateCode(NBlock &root) {
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
  ReturnInst::Create(context, currentBlock());
  popBlock();
}

/* Print the LLVM IR to the given output stream */
void CodeGenContext::printIR(raw_ostream &os) { module->print(os, nullptr); }

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

  auto *MainFn = MainSym->toPtr<void (*)()>();
  MainFn();

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
  return ConstantInt::get(Type::getInt64Ty(context.context), value, true);
}

Value *NDouble::codeGen(CodeGenContext &context) {
  return ConstantFP::get(Type::getDoubleTy(context.context), value);
}

Value *NIdentifier::codeGen(CodeGenContext &context) {
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
  return call;
}

Value *NBinaryOperator::codeGen(CodeGenContext &context) {
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
  case TCEQ:
  case TCNE:
  case TCLT:
  case TCLE:
  case TCGT:
  case TCGE: {
    CmpInst::Predicate pred;
    switch (op) {
    case TCEQ:
      pred = CmpInst::ICMP_EQ;
      break;
    case TCNE:
      pred = CmpInst::ICMP_NE;
      break;
    case TCLT:
      pred = CmpInst::ICMP_SLT;
      break;
    case TCLE:
      pred = CmpInst::ICMP_SLE;
      break;
    case TCGT:
      pred = CmpInst::ICMP_SGT;
      break;
    case TCGE:
      pred = CmpInst::ICMP_SGE;
      break;
    default:
      return NULL;
    }
    Value *cmp =
        CmpInst::Create(Instruction::ICmp, pred, lhs.codeGen(context),
                        rhs.codeGen(context), "", context.currentBlock());
    // Convert i1 to i64 for consistency with other integer operations
    return new ZExtInst(cmp, Type::getInt64Ty(context.context), "",
                        context.currentBlock());
  }
  }

  return NULL;
math:
  return BinaryOperator::Create(instr, lhs.codeGen(context),
                                rhs.codeGen(context), "",
                                context.currentBlock());
}

Value *NAssignment::codeGen(CodeGenContext &context) {
  if (context.locals().find(lhs.name) == context.locals().end()) {
    std::cerr << "undeclared variable " << lhs.name << std::endl;
    return NULL;
  }
  Value *value = rhs.codeGen(context);
  return new StoreInst(value, context.locals()[lhs.name], false,
                       context.currentBlock());
}

Value *NBlock::codeGen(CodeGenContext &context) {
  StatementList::const_iterator it;
  Value *last = NULL;
  for (it = statements.begin(); it != statements.end(); it++) {
    last = (**it).codeGen(context);
  }
  return last;
}

Value *NExpressionStatement::codeGen(CodeGenContext &context) {
  return expression.codeGen(context);
}

Value *NVariableDeclaration::codeGen(CodeGenContext &context) {
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

  // Create allocas for arguments and store the incoming argument values
  Function::arg_iterator argValues = function->arg_begin();
  for (it = arguments.begin(); it != arguments.end(); it++, argValues++) {
    (**it).codeGen(context); // Creates the alloca
    // Store the function argument value into the alloca
    new StoreInst(&*argValues, context.locals()[(**it).id.name], false, bblock);
  }

  // Generate code for the function body and return its value
  Value *retVal = block.codeGen(context);
  ReturnInst::Create(context.context, retVal, context.currentBlock());

  context.popBlock();
  return function;
}

Value *NIfExpression::codeGen(CodeGenContext &context) {
  // Generate code for the condition
  Value *condValue = condition.codeGen(context);
  if (!condValue) {
    return NULL;
  }

  // Convert condition to boolean (compare with 0)
  Value *condBool =
      CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_NE, condValue,
                      ConstantInt::get(Type::getInt64Ty(context.context), 0),
                      "ifcond", context.currentBlock());

  // Get the current function
  Function *func = context.currentBlock()->getParent();

  // Create basic blocks for then, else, and merge
  BasicBlock *thenBB = BasicBlock::Create(context.context, "then", func);
  BasicBlock *elseBB = BasicBlock::Create(context.context, "else", func);
  BasicBlock *mergeBB = BasicBlock::Create(context.context, "ifcont", func);

  // Create conditional branch
  BranchInst::Create(thenBB, elseBB, condBool, context.currentBlock());

  // Emit then block
  context.setCurrentBlock(thenBB);
  Value *thenValue = thenExpr.codeGen(context);
  if (!thenValue) {
    return NULL;
  }
  BranchInst::Create(mergeBB, context.currentBlock());
  // Save the block that then ends in (might be different due to nested ifs)
  BasicBlock *thenEndBB = context.currentBlock();

  // Emit else block
  context.setCurrentBlock(elseBB);
  Value *elseValue = elseExpr.codeGen(context);
  if (!elseValue) {
    return NULL;
  }
  BranchInst::Create(mergeBB, context.currentBlock());
  // Save the block that else ends in
  BasicBlock *elseEndBB = context.currentBlock();

  // Emit merge block with PHI node
  context.setCurrentBlock(mergeBB);
  PHINode *phi =
      PHINode::Create(Type::getInt64Ty(context.context), 2, "iftmp", mergeBB);
  phi->addIncoming(thenValue, thenEndBB);
  phi->addIncoming(elseValue, elseEndBB);

  return phi;
}
