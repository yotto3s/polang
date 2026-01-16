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
static Type* typeOf(const NIdentifier* type, LLVMContext& ctx) noexcept {
  if (type == nullptr) {
    std::cerr << "Internal error: type is null in codegen" << "\n";
    return Type::getVoidTy(ctx);
  }
  if (type->name.compare("int") == 0) {
    return Type::getInt64Ty(ctx);
  } else if (type->name.compare("double") == 0) {
    return Type::getDoubleTy(ctx);
  } else if (type->name.compare("bool") == 0) {
    return Type::getInt1Ty(ctx);
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

void CodeGenVisitor::visit(const NBoolean& node) {
  result_ =
      ConstantInt::get(Type::getInt1Ty(context_.context), node.value ? 1 : 0);
}

void CodeGenVisitor::visit(const NIdentifier& node) {
  if (context_.locals().find(node.name) == context_.locals().end()) {
    std::cerr << "undeclared variable " << node.name << "\n";
    result_ = nullptr;
    return;
  }
  result_ = new LoadInst(context_.locals()[node.name]->getAllocatedType(),
                         context_.locals()[node.name], "", false,
                         context_.currentBlock());
}

void CodeGenVisitor::visit(const NMethodCall& node) {
  Function* const function = context_.module->getFunction(node.id.name.c_str());
  if (function == nullptr) {
    std::cerr << "no such function " << node.id.name << "\n";
    result_ = nullptr;
    return;
  }
  std::vector<Value*> args;

  // Regular arguments
  for (const auto* arg : node.arguments) {
    args.push_back(generate(*arg));
  }

  // Pass captured variables as extra arguments
  const auto& captures = context_.getFunctionCaptures(node.id.name);
  for (const auto& capture_name : captures) {
    if (context_.locals().find(capture_name) == context_.locals().end()) {
      std::cerr << "captured variable " << capture_name
                << " not in scope at call site\n";
      result_ = nullptr;
      return;
    }
    // Load the captured variable's value and pass it
    Value* const captured_value = new LoadInst(
        context_.locals()[capture_name]->getAllocatedType(),
        context_.locals()[capture_name], "", false, context_.currentBlock());
    args.push_back(captured_value);
  }

  result_ = CallInst::Create(function->getFunctionType(), function, args, "",
                             context_.currentBlock());
}

void CodeGenVisitor::visit(const NBinaryOperator& node) {
  Value* const lhs = generate(node.lhs);
  Value* const rhs = generate(node.rhs);
  const bool isDouble = lhs->getType()->isDoubleTy();

  switch (node.op) {
  case TPLUS:
  case TMINUS:
  case TMUL:
  case TDIV: {
    Instruction::BinaryOps instr;
    if (isDouble) {
      switch (node.op) {
      case TPLUS:
        instr = Instruction::FAdd;
        break;
      case TMINUS:
        instr = Instruction::FSub;
        break;
      case TMUL:
        instr = Instruction::FMul;
        break;
      case TDIV:
        instr = Instruction::FDiv;
        break;
      default:
        result_ = nullptr;
        return;
      }
    } else {
      switch (node.op) {
      case TPLUS:
        instr = Instruction::Add;
        break;
      case TMINUS:
        instr = Instruction::Sub;
        break;
      case TMUL:
        instr = Instruction::Mul;
        break;
      case TDIV:
        instr = Instruction::SDiv;
        break;
      default:
        result_ = nullptr;
        return;
      }
    }
    result_ =
        BinaryOperator::Create(instr, lhs, rhs, "", context_.currentBlock());
    return;
  }
  case TCEQ:
  case TCNE:
  case TCLT:
  case TCLE:
  case TCGT:
  case TCGE: {
    if (isDouble) {
      CmpInst::Predicate pred;
      switch (node.op) {
      case TCEQ:
        pred = CmpInst::FCMP_OEQ;
        break;
      case TCNE:
        pred = CmpInst::FCMP_ONE;
        break;
      case TCLT:
        pred = CmpInst::FCMP_OLT;
        break;
      case TCLE:
        pred = CmpInst::FCMP_OLE;
        break;
      case TCGT:
        pred = CmpInst::FCMP_OGT;
        break;
      case TCGE:
        pred = CmpInst::FCMP_OGE;
        break;
      default:
        result_ = nullptr;
        return;
      }
      result_ = CmpInst::Create(Instruction::FCmp, pred, lhs, rhs, "",
                                context_.currentBlock());
    } else {
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
      result_ = CmpInst::Create(Instruction::ICmp, pred, lhs, rhs, "",
                                context_.currentBlock());
    }
    return;
  }
  default:
    result_ = nullptr;
    return;
  }
}

void CodeGenVisitor::visit(const NAssignment& node) {
  if (context_.locals().find(node.lhs.name) == context_.locals().end()) {
    std::cerr << "undeclared variable " << node.lhs.name << "\n";
    result_ = nullptr;
    return;
  }
  Value* const value = generate(node.rhs);
  new StoreInst(value, context_.locals()[node.lhs.name], false,
                context_.currentBlock());
  result_ = value;
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
  if (node.type == nullptr) {
    std::cerr << "Internal error: variable type not resolved for "
              << node.id.name << "\n";
    result_ = nullptr;
    return;
  }
  AllocaInst* const alloc =
      new AllocaInst(typeOf(node.type, context_.context), 0,
                     node.id.name.c_str(), context_.currentBlock());
  context_.locals()[node.id.name] = alloc;
  if (node.assignmentExpr != nullptr) {
    // Inline assignment logic instead of creating temporary NAssignment
    Value* const value = generate(*node.assignmentExpr);
    new StoreInst(value, alloc, false, context_.currentBlock());
  }
  result_ = alloc;
}

void CodeGenVisitor::visit(const NFunctionDeclaration& node) {
  if (node.type == nullptr) {
    std::cerr << "Internal error: function return type not resolved for "
              << node.id.name << "\n";
    result_ = nullptr;
    return;
  }

  // Build argument types: regular params + captured variables
  std::vector<Type*> argTypes;
  for (const auto* arg : node.arguments) {
    if (arg->type == nullptr) {
      std::cerr << "Internal error: parameter type not resolved" << "\n";
      result_ = nullptr;
      return;
    }
    argTypes.push_back(typeOf(arg->type, context_.context));
  }

  // Add captured variable types
  std::vector<std::string> capture_names;
  for (const auto* capture : node.captures) {
    if (capture->type == nullptr) {
      std::cerr << "Internal error: captured variable type not resolved"
                << "\n";
      result_ = nullptr;
      return;
    }
    argTypes.push_back(typeOf(capture->type, context_.context));
    capture_names.push_back(capture->id.name);
  }

  // Store capture info for use when calling this function
  context_.setFunctionCaptures(node.id.name, capture_names);

  FunctionType* const ftype =
      FunctionType::get(typeOf(node.type, context_.context), argTypes, false);
  Function* const function =
      Function::Create(ftype, GlobalValue::InternalLinkage,
                       node.id.name.c_str(), context_.module);
  BasicBlock* const bblock =
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

  // Create allocas for captured variables and store their values
  for (const auto* capture : node.captures) {
    capture->accept(*this); // Creates the alloca
    // Store the captured argument value into the alloca
    new StoreInst(&*argValues, context_.locals()[capture->id.name], false,
                  bblock);
    ++argValues;
  }

  // Generate code for the function body and return its value
  Value* const retVal = generate(node.block);
  ReturnInst::Create(context_.context, retVal, context_.currentBlock());

  context_.popBlock();
  result_ = function;
}

void CodeGenVisitor::visit(const NIfExpression& node) {
  // Generate code for the condition
  Value* const condValue = generate(node.condition);
  if (!condValue) {
    result_ = nullptr;
    return;
  }

  // Condition should be i1 (bool). If it's i64 (legacy int), convert it.
  Value* condBool = condValue;
  if (condValue->getType()->isIntegerTy(64)) {
    condBool =
        CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_NE, condValue,
                        ConstantInt::get(Type::getInt64Ty(context_.context), 0),
                        "ifcond", context_.currentBlock());
  }

  // Get the current function
  Function* const func = context_.currentBlock()->getParent();

  // Create basic blocks for then, else, and merge
  BasicBlock* const thenBB = BasicBlock::Create(context_.context, "then", func);
  BasicBlock* const elseBB = BasicBlock::Create(context_.context, "else", func);
  BasicBlock* const mergeBB =
      BasicBlock::Create(context_.context, "ifcont", func);

  // Create conditional branch
  BranchInst::Create(thenBB, elseBB, condBool, context_.currentBlock());

  // Emit then block
  context_.setCurrentBlock(thenBB);
  Value* const thenValue = generate(node.thenExpr);
  if (!thenValue) {
    result_ = nullptr;
    return;
  }
  BranchInst::Create(mergeBB, context_.currentBlock());
  // Save the block that then ends in (might be different due to nested ifs)
  BasicBlock* const thenEndBB = context_.currentBlock();

  // Emit else block
  context_.setCurrentBlock(elseBB);
  Value* const elseValue = generate(node.elseExpr);
  if (!elseValue) {
    result_ = nullptr;
    return;
  }
  BranchInst::Create(mergeBB, context_.currentBlock());
  // Save the block that else ends in
  BasicBlock* const elseEndBB = context_.currentBlock();

  // Emit merge block with PHI node - use actual type from then value
  context_.setCurrentBlock(mergeBB);
  PHINode* const phi =
      PHINode::Create(thenValue->getType(), 2, "iftmp", mergeBB);
  phi->addIncoming(thenValue, thenEndBB);
  phi->addIncoming(elseValue, elseEndBB);

  result_ = phi;
}

void CodeGenVisitor::visit(const NLetExpression& node) {
  // Save current locals to restore later (scoping)
  const auto savedLocals = context_.locals();

  // Generate code for each binding (can be variables or functions)
  for (const auto* binding : node.bindings) {
    if (binding->isFunction) {
      // Generate the function (will be added to module)
      binding->func->accept(*this);
    } else {
      // Handle variable binding
      const auto* var = binding->var;
      if (var->type == nullptr) {
        std::cerr << "Internal error: let binding type not resolved for "
                  << var->id.name << "\n";
        result_ = nullptr;
        return;
      }
      AllocaInst* const alloc =
          new AllocaInst(typeOf(var->type, context_.context), 0,
                         var->id.name.c_str(), context_.currentBlock());
      context_.locals()[var->id.name] = alloc;

      if (var->assignmentExpr != nullptr) {
        Value* const value = generate(*var->assignmentExpr);
        new StoreInst(value, alloc, false, context_.currentBlock());
      }
    }
  }

  // Generate the body expression
  Value* const bodyValue = generate(node.body);

  // Restore the original locals (remove bindings from scope)
  context_.locals() = savedLocals;

  result_ = bodyValue;
}
