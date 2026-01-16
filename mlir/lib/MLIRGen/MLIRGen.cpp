//===- MLIRGen.cpp - MLIR Generation from Polang AST ------------*- C++ -*-===//
//
// This file implements MLIR generation from the Polang AST.
//
//===----------------------------------------------------------------------===//

#include "polang/MLIRGen.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangTypes.h"

// clang-format off
#include "parser/node.hpp"
#include "parser.hpp"  // Must be after node.hpp for bison union types
// clang-format on
#include "parser/type_checker.hpp"
#include "parser/visitor.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace polang;

namespace {

//===----------------------------------------------------------------------===//
// MLIRGenVisitor - Generates MLIR from Polang AST
//===----------------------------------------------------------------------===//

class MLIRGenVisitor : public Visitor {
public:
  MLIRGenVisitor(MLIRContext &context) : builder(&context) {
    // Create a new module
    module = ModuleOp::create(builder.getUnknownLoc());
  }

  /// Generate MLIR for the given AST block
  ModuleOp generate(const NBlock &block) {
    // First, run type checker to get type information
    typeChecker.check(block);

    // Generate the main function that wraps the top-level code
    generateMainFunction(block);

    // Verify the module
    if (failed(verify(module))) {
      module.emitError("module verification failed");
      return nullptr;
    }

    return module;
  }

  // Visitor interface implementations
  void visit(const NInteger &node) override {
    result = builder.create<ConstantIntOp>(loc(), node.value);
    resultType = "int";
  }

  void visit(const NDouble &node) override {
    result = builder.create<ConstantDoubleOp>(loc(), node.value);
    resultType = "double";
  }

  void visit(const NBoolean &node) override {
    result = builder.create<ConstantBoolOp>(loc(), node.value);
    resultType = "bool";
  }

  void visit(const NIdentifier &node) override {
    // Look up the variable in the symbol table
    auto it = symbolTable.find(node.name);
    if (it != symbolTable.end()) {
      // Load from the alloca
      Value allocaOp = it->second;
      result = builder.create<LoadOp>(loc(), getTypeForName(node.name), allocaOp);
      resultType = typeTable[node.name];
    } else {
      // Check function arguments
      auto argIt = argValues.find(node.name);
      if (argIt != argValues.end()) {
        result = argIt->second;
        resultType = typeTable[node.name];
      } else {
        llvm::errs() << "Unknown variable: " << node.name << "\n";
        result = nullptr;
      }
    }
  }

  void visit(const NMethodCall &node) override {
    // Collect arguments
    SmallVector<Value> args;
    for (const auto *arg : node.arguments) {
      arg->accept(*this);
      if (!result)
        return;
      args.push_back(result);
    }

    // Look up function to get captured variables
    auto funcIt = functionCaptures.find(node.id.name);
    if (funcIt != functionCaptures.end()) {
      // Add captured variables as extra arguments
      for (const auto &captureName : funcIt->second) {
        auto it = symbolTable.find(captureName);
        if (it != symbolTable.end()) {
          Value loadedVal = builder.create<LoadOp>(
              loc(), getTypeForName(captureName), it->second);
          args.push_back(loadedVal);
        } else {
          auto argIt = argValues.find(captureName);
          if (argIt != argValues.end()) {
            args.push_back(argIt->second);
          }
        }
      }
    }

    // Get result type
    auto funcRetTypeIt = functionReturnTypes.find(node.id.name);
    Type resultTy;
    if (funcRetTypeIt != functionReturnTypes.end()) {
      resultTy = getPolangType(funcRetTypeIt->second);
      resultType = funcRetTypeIt->second;
    } else {
      resultTy = builder.getType<IntType>();
      resultType = "int";
    }

    auto callOp = builder.create<CallOp>(loc(), node.id.name,
                                          TypeRange{resultTy}, args);
    result = callOp.getResult();
  }

  void visit(const NBinaryOperator &node) override {
    node.lhs.accept(*this);
    if (!result)
      return;
    Value lhs = result;
    std::string lhsType = resultType;

    node.rhs.accept(*this);
    if (!result)
      return;
    Value rhs = result;

    // Arithmetic operations
    if (node.op == TPLUS) {
      result = builder.create<AddOp>(loc(), lhs, rhs);
      resultType = lhsType;
    } else if (node.op == TMINUS) {
      result = builder.create<SubOp>(loc(), lhs, rhs);
      resultType = lhsType;
    } else if (node.op == TMUL) {
      result = builder.create<MulOp>(loc(), lhs, rhs);
      resultType = lhsType;
    } else if (node.op == TDIV) {
      result = builder.create<DivOp>(loc(), lhs, rhs);
      resultType = lhsType;
    }
    // Comparison operations
    else if (node.op == TCEQ) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::eq, lhs, rhs);
      resultType = "bool";
    } else if (node.op == TCNE) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::ne, lhs, rhs);
      resultType = "bool";
    } else if (node.op == TCLT) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::lt, lhs, rhs);
      resultType = "bool";
    } else if (node.op == TCLE) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::le, lhs, rhs);
      resultType = "bool";
    } else if (node.op == TCGT) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::gt, lhs, rhs);
      resultType = "bool";
    } else if (node.op == TCGE) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::ge, lhs, rhs);
      resultType = "bool";
    } else {
      llvm::errs() << "Unknown binary operator: " << node.op << "\n";
      result = nullptr;
    }
  }

  void visit(const NAssignment &node) override {
    // Evaluate RHS
    node.rhs.accept(*this);
    if (!result)
      return;
    Value value = result;

    // Get the alloca for the variable
    auto it = symbolTable.find(node.lhs.name);
    if (it == symbolTable.end()) {
      llvm::errs() << "Unknown variable in assignment: " << node.lhs.name << "\n";
      result = nullptr;
      return;
    }

    // Store the value
    builder.create<StoreOp>(loc(), value, it->second);

    // Assignment expression returns the assigned value
    result = value;
  }

  void visit(const NBlock &node) override {
    Value lastValue = nullptr;

    for (const auto *stmt : node.statements) {
      stmt->accept(*this);
      if (result)
        lastValue = result;
    }

    result = lastValue;
  }

  void visit(const NIfExpression &node) override {
    // Evaluate condition
    node.condition.accept(*this);
    if (!result)
      return;
    Value condition = result;

    // Infer result type from the type checker
    node.thenExpr.accept(*this);
    std::string ifResultType = resultType;

    // Create if operation
    Type resultTy = getPolangType(ifResultType);
    auto ifOp = builder.create<IfOp>(loc(), resultTy, condition);

    // Generate then region
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      node.thenExpr.accept(*this);
      if (result)
        builder.create<YieldOp>(loc(), result);
    }

    // Generate else region
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      node.elseExpr.accept(*this);
      if (result)
        builder.create<YieldOp>(loc(), result);
    }

    result = ifOp.getResult();
    resultType = ifResultType;
  }

  void visit(const NLetExpression &node) override {
    // Save current symbol table state
    auto savedSymbolTable = symbolTable;
    auto savedTypeTable = typeTable;

    // Process bindings
    for (const auto *binding : node.bindings) {
      if (binding->isFunction) {
        binding->func->accept(*this);
      } else {
        binding->var->accept(*this);
      }
    }

    // Evaluate body
    node.body.accept(*this);

    // Restore symbol table (let bindings are scoped)
    symbolTable = savedSymbolTable;
    typeTable = savedTypeTable;
  }

  void visit(const NExpressionStatement &node) override {
    node.expression.accept(*this);
  }

  void visit(const NVariableDeclaration &node) override {
    // Determine the type
    std::string typeName = node.type ? node.type->name : "int";
    if (node.assignmentExpr) {
      // Infer type from assignment
      node.assignmentExpr->accept(*this);
      typeName = resultType;
    }

    Type polangType = getPolangType(typeName);
    Type llvmType = convertPolangType(polangType);

    // Create alloca using the correct builder signature
    auto memRefType = MemRefType::get({}, llvmType);
    auto alloca = builder.create<AllocaOp>(
        loc(), memRefType, node.id.name, polangType, node.isMutable);

    // Store initial value if present
    if (node.assignmentExpr && result) {
      builder.create<StoreOp>(loc(), result, alloca);
    }

    // Register in symbol table
    symbolTable[node.id.name] = alloca;
    typeTable[node.id.name] = typeName;
  }

  void visit(const NFunctionDeclaration &node) override {
    // Save the current insertion point
    OpBuilder::InsertionGuard guard(builder);

    // Build function type
    SmallVector<Type> argTypes;
    std::vector<std::string> argNames;

    for (const auto *arg : node.arguments) {
      std::string typeName = arg->type ? arg->type->name : "int";
      argTypes.push_back(getPolangType(typeName));
      argNames.push_back(arg->id.name);
    }

    // Add captured variables as extra parameters
    std::vector<std::string> captureNames;
    for (const auto *capture : node.captures) {
      std::string typeName = capture->type ? capture->type->name : "int";
      argTypes.push_back(getPolangType(typeName));
      captureNames.push_back(capture->id.name);
    }

    // Store captures for call site
    functionCaptures[node.id.name] = captureNames;

    // Return type
    std::string returnTypeName = node.type ? node.type->name : "int";
    Type returnType = getPolangType(returnTypeName);
    functionReturnTypes[node.id.name] = returnTypeName;

    auto funcType = builder.getFunctionType(argTypes, {returnType});

    // Create function at module level
    builder.setInsertionPointToEnd(module.getBody());

    // Convert captureNames to ArrayRef<StringRef>
    SmallVector<StringRef> captureRefs;
    for (const auto &name : captureNames) {
      captureRefs.push_back(name);
    }

    auto funcOp = builder.create<FuncOp>(loc(), node.id.name, funcType,
                                          ArrayRef<StringRef>(captureRefs));

    // Create entry block with arguments
    Block *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Save current state
    auto savedSymbolTable = symbolTable;
    auto savedTypeTable = typeTable;
    auto savedArgValues = argValues;

    symbolTable.clear();
    typeTable.clear();
    argValues.clear();

    // Register function arguments
    size_t argIdx = 0;
    for (const auto *arg : node.arguments) {
      std::string typeName = arg->type ? arg->type->name : "int";
      argValues[arg->id.name] = entryBlock->getArgument(argIdx);
      typeTable[arg->id.name] = typeName;
      ++argIdx;
    }

    // Register captured variables as arguments
    for (const auto *capture : node.captures) {
      std::string typeName = capture->type ? capture->type->name : "int";
      argValues[capture->id.name] = entryBlock->getArgument(argIdx);
      typeTable[capture->id.name] = typeName;
      ++argIdx;
    }

    // Generate function body
    node.block.accept(*this);

    // Add return
    if (result) {
      builder.create<ReturnOp>(loc(), result);
    } else {
      builder.create<ReturnOp>(loc());
    }

    // Restore state
    symbolTable = savedSymbolTable;
    typeTable = savedTypeTable;
    argValues = savedArgValues;

    result = nullptr; // Function declarations don't produce a value
  }

private:
  OpBuilder builder;
  ModuleOp module;
  TypeChecker typeChecker;

  // Current result value and type
  Value result;
  std::string resultType;

  // Symbol tables
  std::map<std::string, Value> symbolTable;  // Variable allocas
  std::map<std::string, Value> argValues;    // Function arguments
  std::map<std::string, std::string> typeTable;  // Variable types
  std::map<std::string, std::vector<std::string>> functionCaptures;
  std::map<std::string, std::string> functionReturnTypes;

  Location loc() { return builder.getUnknownLoc(); }

  Type getPolangType(const std::string &typeName) {
    if (typeName == "int")
      return builder.getType<IntType>();
    if (typeName == "double")
      return builder.getType<DoubleType>();
    if (typeName == "bool")
      return builder.getType<BoolType>();
    // Default to int
    return builder.getType<IntType>();
  }

  Type getTypeForName(const std::string &name) {
    auto it = typeTable.find(name);
    if (it != typeTable.end()) {
      return getPolangType(it->second);
    }
    return builder.getType<IntType>();
  }

  Type convertPolangType(Type polangType) {
    if (isa<IntType>(polangType))
      return builder.getI64Type();
    if (isa<DoubleType>(polangType))
      return builder.getF64Type();
    if (isa<BoolType>(polangType))
      return builder.getI1Type();
    return builder.getI64Type();
  }

  void generateMainFunction(const NBlock &block) {
    // Get the inferred return type from the type checker (already ran in generate())
    std::string inferredType = typeChecker.getInferredType();
    Type returnType = getPolangType(inferredType);

    // Create entry function with dynamic return type
    auto funcType = builder.getFunctionType({}, {returnType});

    builder.setInsertionPointToEnd(module.getBody());
    auto entryFunc = builder.create<FuncOp>(loc(), "__polang_entry", funcType);

    // Create entry block
    Block *entryBlock = entryFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate code for the block
    block.accept(*this);

    // Return the last expression value, or default value of correct type
    if (result) {
      builder.create<ReturnOp>(loc(), result);
    } else {
      // Create default value matching the return type
      Value defaultVal;
      if (inferredType == "double") {
        defaultVal = builder.create<ConstantDoubleOp>(loc(), 0.0);
      } else if (inferredType == "bool") {
        defaultVal = builder.create<ConstantBoolOp>(loc(), false);
      } else {
        defaultVal = builder.create<ConstantIntOp>(loc(), 0);
      }
      builder.create<ReturnOp>(loc(), defaultVal);
    }
  }
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> polang::mlirGen(mlir::MLIRContext &context,
                                                   const NBlock &moduleAST) {
  // Register the Polang dialect
  context.getOrLoadDialect<PolangDialect>();

  MLIRGenVisitor generator(context);
  ModuleOp module = generator.generate(moduleAST);
  if (!module)
    return nullptr;

  return module;
}
