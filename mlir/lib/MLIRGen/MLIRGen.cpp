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
#include "parser/polang_types.hpp"
#include "parser/type_checker.hpp"
#include "parser/visitor.hpp"

using polang::TypeNames;

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
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
  MLIRGenVisitor(MLIRContext& context, bool emitTypeVars = false)
      : builder(&context), emitTypeVars_(emitTypeVars) {
    // Create a new module
    module = ModuleOp::create(builder.getUnknownLoc());
  }

  /// Generate a fresh type variable
  Type freshTypeVar() {
    return builder.getType<TypeVarType>(nextTypeVarId_++);
  }

  /// Get a Polang type from annotation, or a fresh type variable if none
  Type getTypeOrFresh(const NIdentifier* typeAnnotation) {
    if (typeAnnotation) {
      return getPolangType(typeAnnotation->name);
    }
    // No annotation - always emit type variable for polymorphic inference
    return freshTypeVar();
  }

  /// Generate MLIR for the given AST block
  ModuleOp generate(const NBlock& block) {
    // Always run type checker for error detection (undefined vars, etc.)
    typeChecker.check(block);

    // If type checker found errors, fail early
    if (typeChecker.hasErrors()) {
      return nullptr;
    }

    // Generate the main function that wraps the top-level code
    generateMainFunction(block);

    // Skip module verification since type variables may be present.
    // Type inference pass will resolve them and verification will happen later.
    // We keep verification disabled to allow polymorphic types.
    if (false) { // Disabled: verification happens after type inference
      if (failed(verify(module))) {
        module.emitError("module verification failed");
        return nullptr;
      }
    }

    return module;
  }

  // Visitor interface implementations
  void visit(const NInteger& node) override {
    result = builder.create<ConstantIntOp>(loc(), node.value);
    resultType = TypeNames::INT;
  }

  void visit(const NDouble& node) override {
    result = builder.create<ConstantDoubleOp>(loc(), node.value);
    resultType = TypeNames::DOUBLE;
  }

  void visit(const NBoolean& node) override {
    result = builder.create<ConstantBoolOp>(loc(), node.value);
    resultType = TypeNames::BOOL;
  }

  void visit(const NIdentifier& node) override {
    auto value = lookupVariable(node.name);
    if (value) {
      result = *value;
      auto type = lookupType(node.name);
      resultType = type.value_or(TypeNames::INT);
      return;
    }

    emitError(loc()) << "Unknown variable: " << node.name;
    result = nullptr;
  }

  void visit(const NMethodCall& node) override {
    // Collect arguments
    SmallVector<Value> args;
    for (const auto* arg : node.arguments) {
      arg->accept(*this);
      if (!result)
        return;
      args.push_back(result);
    }

    // Look up function to get captured variables
    auto funcIt = functionCaptures.find(node.id.name);
    if (funcIt != functionCaptures.end()) {
      // Add captured variables as extra arguments
      for (const auto& captureName : funcIt->second) {
        auto capturedValue = lookupVariable(captureName);
        if (capturedValue) {
          args.push_back(*capturedValue);
        }
      }
    }

    // Get result type - prefer MLIR type (may include type vars)
    Type resultTy;
    auto funcRetMLIRIt = functionReturnMLIRTypes.find(node.id.name);
    if (funcRetMLIRIt != functionReturnMLIRTypes.end()) {
      resultTy = funcRetMLIRIt->second;
      // Set resultType string if we have it
      auto funcRetTypeIt = functionReturnTypes.find(node.id.name);
      if (funcRetTypeIt != functionReturnTypes.end()) {
        resultType = funcRetTypeIt->second;
      } else {
        resultType = TypeNames::UNKNOWN;
      }
    } else {
      // Fallback to string-based lookup
      auto funcRetTypeIt = functionReturnTypes.find(node.id.name);
      if (funcRetTypeIt != functionReturnTypes.end()) {
        resultTy = getPolangType(funcRetTypeIt->second);
        resultType = funcRetTypeIt->second;
      } else {
        // Function not found - generate fresh type var for unknown function
        resultTy = freshTypeVar();
        resultType = TypeNames::UNKNOWN;
      }
    }

    auto callOp =
        builder.create<CallOp>(loc(), node.id.name, TypeRange{resultTy}, args);
    result = callOp.getResult();
  }

  void visit(const NBinaryOperator& node) override {
    node.lhs.accept(*this);
    if (!result)
      return;
    Value lhs = result;
    std::string lhsType = resultType;

    node.rhs.accept(*this);
    if (!result)
      return;
    Value rhs = result;

    // Arithmetic operations - use LHS type as result type
    // This allows type variables since the verifier uses typesAreCompatible()
    Type arithResultTy = lhs.getType();
    if (node.op == TPLUS) {
      result = builder.create<AddOp>(loc(), arithResultTy, lhs, rhs);
      resultType = lhsType;
    } else if (node.op == TMINUS) {
      result = builder.create<SubOp>(loc(), arithResultTy, lhs, rhs);
      resultType = lhsType;
    } else if (node.op == TMUL) {
      result = builder.create<MulOp>(loc(), arithResultTy, lhs, rhs);
      resultType = lhsType;
    } else if (node.op == TDIV) {
      result = builder.create<DivOp>(loc(), arithResultTy, lhs, rhs);
      resultType = lhsType;
    }
    // Comparison operations
    else if (node.op == TCEQ) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::eq, lhs, rhs);
      resultType = TypeNames::BOOL;
    } else if (node.op == TCNE) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::ne, lhs, rhs);
      resultType = TypeNames::BOOL;
    } else if (node.op == TCLT) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::lt, lhs, rhs);
      resultType = TypeNames::BOOL;
    } else if (node.op == TCLE) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::le, lhs, rhs);
      resultType = TypeNames::BOOL;
    } else if (node.op == TCGT) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::gt, lhs, rhs);
      resultType = TypeNames::BOOL;
    } else if (node.op == TCGE) {
      result = builder.create<CmpOp>(loc(), CmpPredicate::ge, lhs, rhs);
      resultType = TypeNames::BOOL;
    } else {
      emitError(loc()) << "Unknown binary operator: " << node.op;
      result = nullptr;
    }
  }

  void visit(const NAssignment& node) override {
    // Evaluate RHS
    node.rhs.accept(*this);
    if (!result)
      return;
    Value value = result;

    // Get the alloca for the variable
    auto it = symbolTable.find(node.lhs.name);
    if (it == symbolTable.end()) {
      emitError(loc()) << "Unknown variable in assignment: " << node.lhs.name;
      result = nullptr;
      return;
    }

    // Store the value
    builder.create<StoreOp>(loc(), value, it->second);

    // Assignment expression returns the assigned value
    result = value;
  }

  void visit(const NBlock& node) override {
    Value lastValue = nullptr;

    for (const auto* stmt : node.statements) {
      stmt->accept(*this);
      if (result)
        lastValue = result;
    }

    result = lastValue;
  }

  void visit(const NIfExpression& node) override {
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

  void visit(const NLetExpression& node) override {
    // RAII scope guard automatically saves/restores symbol tables
    SymbolTableScope scope(*this);

    // Process bindings
    for (const auto* binding : node.bindings) {
      if (binding->isFunction) {
        binding->func->accept(*this);
      } else {
        binding->var->accept(*this);
      }
    }

    // Evaluate body
    node.body.accept(*this);
  }

  void visit(const NExpressionStatement& node) override {
    node.expression.accept(*this);
  }

  void visit(const NVariableDeclaration& node) override {
    // Determine the type
    std::string typeName = node.type ? node.type->name : TypeNames::INT;
    if (node.assignmentExpr) {
      // Infer type from assignment
      node.assignmentExpr->accept(*this);
      typeName = resultType;
    }

    if (node.isMutable) {
      // Mutable: use alloca/store pattern
      Type polangType = getPolangType(typeName);
      Type llvmType = convertPolangType(polangType);

      auto memRefType = MemRefType::get({}, llvmType);
      auto alloca = builder.create<AllocaOp>(loc(), memRefType, node.id.name,
                                             polangType, node.isMutable);

      if (node.assignmentExpr && result) {
        builder.create<StoreOp>(loc(), result, alloca);
      }

      symbolTable[node.id.name] = alloca;
    } else {
      // Immutable: store SSA value directly (no alloca needed)
      immutableValues[node.id.name] = result;
    }

    typeTable[node.id.name] = typeName;
  }

  void visit(const NFunctionDeclaration& node) override {
    // Save the current insertion point
    OpBuilder::InsertionGuard guard(builder);

    // Build function type with type variables for untyped parameters
    SmallVector<Type> argTypes;
    std::vector<std::string> argNames;
    std::vector<Type> argMLIRTypes; // Track MLIR types including type vars

    for (const auto* arg : node.arguments) {
      Type argType = getTypeOrFresh(arg->type);
      argTypes.push_back(argType);
      argMLIRTypes.push_back(argType);
      argNames.push_back(arg->id.name);
    }

    // Add captured variables as extra parameters
    std::vector<std::string> captureNames;
    std::vector<Type> captureMLIRTypes;
    for (const auto* capture : node.captures) {
      Type captureType = getTypeOrFresh(capture->type);
      argTypes.push_back(captureType);
      captureMLIRTypes.push_back(captureType);
      captureNames.push_back(capture->id.name);
    }

    // Store captures for call site
    functionCaptures[node.id.name] = captureNames;

    // Return type - use type variable if not specified
    Type returnType = getTypeOrFresh(node.type);
    if (node.type) {
      functionReturnTypes[node.id.name] = node.type->name;
    }
    functionReturnMLIRTypes[node.id.name] = returnType;

    auto funcType = builder.getFunctionType(argTypes, {returnType});

    // Create function at module level
    builder.setInsertionPointToEnd(module.getBody());

    // Convert captureNames to ArrayRef<StringRef>
    SmallVector<StringRef> captureRefs;
    for (const auto& name : captureNames) {
      captureRefs.push_back(name);
    }

    auto funcOp = builder.create<FuncOp>(loc(), node.id.name, funcType,
                                         ArrayRef<StringRef>(captureRefs));

    // Create entry block with arguments
    Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // RAII scope guard clears and restores symbol tables for function body
    SymbolTableScope scope(*this, /*clearAllTables=*/true);

    // Register function arguments with their MLIR types
    size_t argIdx = 0;
    for (size_t i = 0; i < node.arguments.size(); ++i) {
      const auto* arg = node.arguments[i];
      argValues[arg->id.name] = entryBlock->getArgument(argIdx);
      typeVarTable[arg->id.name] = argMLIRTypes[i];
      if (arg->type) {
        typeTable[arg->id.name] = arg->type->name;
      }
      ++argIdx;
    }

    // Register captured variables as arguments
    for (size_t i = 0; i < node.captures.size(); ++i) {
      const auto* capture = node.captures[i];
      argValues[capture->id.name] = entryBlock->getArgument(argIdx);
      typeVarTable[capture->id.name] = captureMLIRTypes[i];
      if (capture->type) {
        typeTable[capture->id.name] = capture->type->name;
      }
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

    result = nullptr; // Function declarations don't produce a value
  }

private:
  OpBuilder builder;
  ModuleOp module;
  TypeChecker typeChecker;

  // Whether to emit type variables for untyped positions (polymorphic mode)
  bool emitTypeVars_;

  // Type variable counter for generating fresh type variables
  uint64_t nextTypeVarId_ = 0;

  // Current result value and type
  Value result;
  std::string resultType;

  // Symbol tables
  std::map<std::string, Value> symbolTable;     // Mutable variable allocas
  std::map<std::string, Value> argValues;       // Function arguments
  std::map<std::string, Value> immutableValues; // Immutable variable SSA values
  std::map<std::string, std::string> typeTable; // Variable types
  std::map<std::string, Type> typeVarTable;     // Variable types as MLIR types (for type vars)
  std::map<std::string, std::vector<std::string>> functionCaptures;
  std::map<std::string, std::string> functionReturnTypes;
  std::map<std::string, Type> functionReturnMLIRTypes; // Function return types as MLIR types

  /// RAII helper class for scoped symbol table management.
  /// Automatically saves and restores symbol tables when entering/exiting
  /// scopes.
  class SymbolTableScope {
  public:
    SymbolTableScope(MLIRGenVisitor& visitor, bool clearAllTables = false)
        : visitor_(visitor), savedSymbolTable_(visitor.symbolTable),
          savedTypeTable_(visitor.typeTable),
          savedTypeVarTable_(visitor.typeVarTable),
          savedArgValues_(visitor.argValues),
          savedImmutableValues_(visitor.immutableValues) {
      if (clearAllTables) {
        visitor_.symbolTable.clear();
        visitor_.typeTable.clear();
        visitor_.typeVarTable.clear();
        visitor_.argValues.clear();
        visitor_.immutableValues.clear();
      }
    }

    ~SymbolTableScope() {
      visitor_.symbolTable = savedSymbolTable_;
      visitor_.typeTable = savedTypeTable_;
      visitor_.typeVarTable = savedTypeVarTable_;
      visitor_.argValues = savedArgValues_;
      visitor_.immutableValues = savedImmutableValues_;
    }

  private:
    MLIRGenVisitor& visitor_;
    std::map<std::string, Value> savedSymbolTable_;
    std::map<std::string, std::string> savedTypeTable_;
    std::map<std::string, Type> savedTypeVarTable_;
    std::map<std::string, Value> savedArgValues_;
    std::map<std::string, Value> savedImmutableValues_;
  };

  Location loc() { return builder.getUnknownLoc(); }

  /// Look up a variable by name in the symbol tables.
  /// Checks immutable values, mutable allocas, and function arguments.
  /// Returns nullopt if the variable is not found.
  std::optional<Value> lookupVariable(const std::string& name) {
    // Check immutable values first (SSA values, no load needed)
    auto immIt = immutableValues.find(name);
    if (immIt != immutableValues.end()) {
      return immIt->second;
    }

    // Check mutable variables (allocas, need load)
    auto it = symbolTable.find(name);
    if (it != symbolTable.end()) {
      return builder.create<LoadOp>(loc(), getTypeForName(name), it->second);
    }

    // Check function arguments
    auto argIt = argValues.find(name);
    if (argIt != argValues.end()) {
      return argIt->second;
    }

    return std::nullopt;
  }

  /// Look up a variable's type by name.
  std::optional<std::string> lookupType(const std::string& name) {
    auto it = typeTable.find(name);
    if (it != typeTable.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  Type getPolangType(const std::string& typeName) {
    if (typeName == TypeNames::INT)
      return builder.getType<IntType>();
    if (typeName == TypeNames::DOUBLE)
      return builder.getType<DoubleType>();
    if (typeName == TypeNames::BOOL)
      return builder.getType<BoolType>();
    if (typeName == TypeNames::TYPEVAR)
      return freshTypeVar();
    // Default to int
    return builder.getType<IntType>();
  }

  Type getTypeForName(const std::string& name) {
    // First check typeVarTable for MLIR types (including type vars)
    auto varIt = typeVarTable.find(name);
    if (varIt != typeVarTable.end()) {
      return varIt->second;
    }
    // Fall back to string-based type table
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

  void generateMainFunction(const NBlock& block) {
    // Get the inferred return type from the type checker (already ran in
    // generate())
    std::string inferredType = typeChecker.getInferredType();
    Type returnType = getPolangType(inferredType);

    // Create entry function with dynamic return type
    auto funcType = builder.getFunctionType({}, {returnType});

    builder.setInsertionPointToEnd(module.getBody());
    auto entryFunc = builder.create<FuncOp>(loc(), "__polang_entry", funcType);

    // Create entry block
    Block* entryBlock = entryFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate code for the block
    block.accept(*this);

    // Return the last expression value, or default value of correct type
    if (result) {
      builder.create<ReturnOp>(loc(), result);
    } else {
      // Create default value matching the return type
      Value defaultVal;
      if (inferredType == TypeNames::DOUBLE) {
        defaultVal = builder.create<ConstantDoubleOp>(loc(), 0.0);
      } else if (inferredType == TypeNames::BOOL) {
        defaultVal = builder.create<ConstantBoolOp>(loc(), false);
      } else {
        defaultVal = builder.create<ConstantIntOp>(loc(), 0);
      }
      builder.create<ReturnOp>(loc(), defaultVal);
    }
  }
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> polang::mlirGen(mlir::MLIRContext& context,
                                                  const NBlock& moduleAST,
                                                  bool emitTypeVars) {
  // Register the Polang dialect
  context.getOrLoadDialect<PolangDialect>();

  MLIRGenVisitor generator(context, emitTypeVars);
  ModuleOp module = generator.generate(moduleAST);
  if (!module)
    return nullptr;

  return module;
}
