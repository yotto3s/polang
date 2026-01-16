#include "compiler/codegen_visitor.hpp"
#include "compiler/codegen.hpp"
// clang-format off
#include "parser/node.hpp"
#include "parser.hpp" // Must be after node.hpp for bison union types
// clang-format on

#include <iostream>

using namespace llvm;

CodeGenVisitor::CodeGenVisitor(CodeGenContext& context) : context_(context) {}

Value* CodeGenVisitor::generate(const Node& node) {
  node.accept(*this);
  return result_;
}

/* Returns an LLVM type based on the identifier */
static Type* typeOf(const NIdentifier& type, LLVMContext& ctx) noexcept {
  if (type.name.compare("int") == 0) {
    return Type::getInt64Ty(ctx);
  } else if (type.name.compare("double") == 0) {
    return Type::getDoubleTy(ctx);
  }
  return Type::getVoidTy(ctx);
}

void CodeGenVisitor::visit(const NInteger& node) {
  result_ =
      ConstantInt::get(Type::getInt64Ty(context_.context), node.value, true);
}

void CodeGenVisitor::visit(const NDouble& node) {
  result_ = ConstantFP::get(Type::getDoubleTy(context_.context), node.value);
}

void CodeGenVisitor::visit(const NIdentifier& node) {
  if (context_.locals().find(node.name) == context_.locals().end()) {
    std::cerr << "undeclared variable " << node.name << std::endl;
    result_ = nullptr;
    return;
  }
  result_ = new LoadInst(context_.locals()[node.name]->getAllocatedType(),
                         context_.locals()[node.name], "", false,
                         context_.currentBlock());
}

void CodeGenVisitor::visit(const NMethodCall& node) {
  Function* function = context_.module->getFunction(node.id.name.c_str());
  if (function == nullptr) {
    std::cerr << "no such function " << node.id.name << std::endl;
    result_ = nullptr;
    return;
  }
  std::vector<Value*> args;
  for (const auto* arg : node.arguments) {
    args.push_back(generate(*arg));
  }
  result_ = CallInst::Create(function->getFunctionType(), function, args, "",
                             context_.currentBlock());
}

void CodeGenVisitor::visit(const NBinaryOperator& node) {
  Instruction::BinaryOps instr;
  switch (node.op) {
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
    switch (node.op) {
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
      result_ = nullptr;
      return;
    }
    Value* cmp =
        CmpInst::Create(Instruction::ICmp, pred, generate(node.lhs),
                        generate(node.rhs), "", context_.currentBlock());
    // Convert i1 to i64 for consistency with other integer operations
    result_ = new ZExtInst(cmp, Type::getInt64Ty(context_.context), "",
                           context_.currentBlock());
    return;
  }
  }

  result_ = nullptr;
  return;
math:
  result_ =
      BinaryOperator::Create(instr, generate(node.lhs), generate(node.rhs), "",
                             context_.currentBlock());
}

void CodeGenVisitor::visit(const NAssignment& node) {
  if (context_.locals().find(node.lhs.name) == context_.locals().end()) {
    std::cerr << "undeclared variable " << node.lhs.name << std::endl;
    result_ = nullptr;
    return;
  }
  Value* value = generate(node.rhs);
  result_ = new StoreInst(value, context_.locals()[node.lhs.name], false,
                          context_.currentBlock());
}

void CodeGenVisitor::visit(const NBlock& node) {
  Value* last = nullptr;
  for (const auto* stmt : node.statements) {
    last = generate(*stmt);
  }
  result_ = last;
}

void CodeGenVisitor::visit(const NExpressionStatement& node) {
  result_ = generate(node.expression);
}

void CodeGenVisitor::visit(const NVariableDeclaration& node) {
  AllocaInst* alloc =
      new AllocaInst(typeOf(node.type, context_.context), 0,
                     node.id.name.c_str(), context_.currentBlock());
  context_.locals()[node.id.name] = alloc;
  if (node.assignmentExpr != nullptr) {
    // Inline assignment logic instead of creating temporary NAssignment
    Value* value = generate(*node.assignmentExpr);
    new StoreInst(value, alloc, false, context_.currentBlock());
  }
  result_ = alloc;
}

void CodeGenVisitor::visit(const NFunctionDeclaration& node) {
  std::vector<Type*> argTypes;
  for (const auto* arg : node.arguments) {
    argTypes.push_back(typeOf(arg->type, context_.context));
  }
  FunctionType* ftype =
      FunctionType::get(typeOf(node.type, context_.context), argTypes, false);
  Function* function = Function::Create(ftype, GlobalValue::InternalLinkage,
                                        node.id.name.c_str(), context_.module);
  BasicBlock* bblock =
      BasicBlock::Create(context_.context, "entry", function, 0);

  context_.pushBlock(bblock);

  // Create allocas for arguments and store the incoming argument values
  Function::arg_iterator argValues = function->arg_begin();
  for (const auto* arg : node.arguments) {
    arg->accept(*this); // Creates the alloca
    // Store the function argument value into the alloca
    new StoreInst(&*argValues, context_.locals()[arg->id.name], false, bblock);
    ++argValues;
  }

  // Generate code for the function body and return its value
  Value* retVal = generate(node.block);
  ReturnInst::Create(context_.context, retVal, context_.currentBlock());

  context_.popBlock();
  result_ = function;
}

void CodeGenVisitor::visit(const NIfExpression& node) {
  // Generate code for the condition
  Value* condValue = generate(node.condition);
  if (!condValue) {
    result_ = nullptr;
    return;
  }

  // Convert condition to boolean (compare with 0)
  Value* condBool =
      CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_NE, condValue,
                      ConstantInt::get(Type::getInt64Ty(context_.context), 0),
                      "ifcond", context_.currentBlock());

  // Get the current function
  Function* func = context_.currentBlock()->getParent();

  // Create basic blocks for then, else, and merge
  BasicBlock* thenBB = BasicBlock::Create(context_.context, "then", func);
  BasicBlock* elseBB = BasicBlock::Create(context_.context, "else", func);
  BasicBlock* mergeBB = BasicBlock::Create(context_.context, "ifcont", func);

  // Create conditional branch
  BranchInst::Create(thenBB, elseBB, condBool, context_.currentBlock());

  // Emit then block
  context_.setCurrentBlock(thenBB);
  Value* thenValue = generate(node.thenExpr);
  if (!thenValue) {
    result_ = nullptr;
    return;
  }
  BranchInst::Create(mergeBB, context_.currentBlock());
  // Save the block that then ends in (might be different due to nested ifs)
  BasicBlock* thenEndBB = context_.currentBlock();

  // Emit else block
  context_.setCurrentBlock(elseBB);
  Value* elseValue = generate(node.elseExpr);
  if (!elseValue) {
    result_ = nullptr;
    return;
  }
  BranchInst::Create(mergeBB, context_.currentBlock());
  // Save the block that else ends in
  BasicBlock* elseEndBB = context_.currentBlock();

  // Emit merge block with PHI node
  context_.setCurrentBlock(mergeBB);
  PHINode* phi =
      PHINode::Create(Type::getInt64Ty(context_.context), 2, "iftmp", mergeBB);
  phi->addIncoming(thenValue, thenEndBB);
  phi->addIncoming(elseValue, elseEndBB);

  result_ = phi;
}

void CodeGenVisitor::visit(const NLetExpression& node) {
  // Save current locals to restore later (scoping)
  const auto savedLocals = context_.locals();

  // Generate code for each binding
  for (const auto* binding : node.bindings) {
    AllocaInst* alloc =
        new AllocaInst(typeOf(binding->type, context_.context), 0,
                       binding->id.name.c_str(), context_.currentBlock());
    context_.locals()[binding->id.name] = alloc;

    if (binding->assignmentExpr != nullptr) {
      Value* value = generate(*binding->assignmentExpr);
      new StoreInst(value, alloc, false, context_.currentBlock());
    }
  }

  // Generate the body expression
  Value* bodyValue = generate(node.body);

  // Restore the original locals (remove bindings from scope)
  context_.locals() = savedLocals;

  result_ = bodyValue;
}
