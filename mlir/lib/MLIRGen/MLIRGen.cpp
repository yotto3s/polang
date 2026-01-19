//===- MLIRGen.cpp - MLIR Generation from Polang AST ------------*- C++ -*-===//
//
// This file implements MLIR generation from the Polang AST.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

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

#pragma GCC diagnostic pop

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
  MLIRGenVisitor(MLIRContext& context, bool /*emitTypeVars*/ = false)
      : builder(&context) {
    // Create a new module
    module = ModuleOp::create(builder.getUnknownLoc());
  }

  /// Generate a fresh type variable with optional kind constraint
  Type freshTypeVar(TypeVarKind kind = TypeVarKind::Any) {
    return builder.getType<TypeVarType>(nextTypeVarId++, kind);
  }

  /// Get a Polang type from annotation, or a fresh type variable if none
  Type getTypeOrFresh(const NIdentifier* typeAnnotation) {
    if (typeAnnotation != nullptr) {
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
    // Note: Verification is intentionally disabled here.

    return module;
  }

  // Visitor interface implementations
  void visit(const NInteger& node) override {
    // Create type variable constrained to integer types
    auto type = freshTypeVar(TypeVarKind::Integer);
    result = builder.create<ConstantIntegerOp>(loc(), type, node.value);
    resultType = TypeNames::GENERIC_INT;
  }

  void visit(const NDouble& node) override {
    // Create type variable constrained to float types
    auto type = freshTypeVar(TypeVarKind::Float);
    result = builder.create<ConstantFloatOp>(loc(), type, node.value);
    resultType = TypeNames::GENERIC_FLOAT;
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
      resultType = type.value_or(TypeNames::I64);
      return;
    }

    emitError(loc()) << "Unknown variable: " << node.name;
    result = nullptr;
  }

  void visit(const NQualifiedName& node) override {
    // Qualified name access (e.g., Math.PI)
    // Look up using mangled name
    const std::string mangled = node.mangledName();
    auto value = lookupVariable(mangled);
    if (value) {
      result = *value;
      auto type = lookupType(mangled);
      resultType = type.value_or(TypeNames::I64);
      return;
    }

    emitError(loc()) << "Unknown qualified name: " << node.fullName();
    result = nullptr;
  }

  void visit(const NMethodCall& node) override {
    // Get the effective function name (mangled for qualified calls)
    std::string funcName = node.effectiveName();

    // Resolve imported symbol to mangled name if applicable
    auto importIt = importedSymbols.find(funcName);
    if (importIt != importedSymbols.end()) {
      funcName = importIt->second;
    }

    // Collect arguments
    SmallVector<Value> args;
    for (const auto& arg : node.arguments) {
      arg->accept(*this);
      if (!result) {
        return;
      }
      args.push_back(result);
    }

    // Look up function to get captured variables
    auto funcIt = functionCaptures.find(funcName);
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
    auto funcRetMLIRIt = functionReturnMLIRTypes.find(funcName);
    if (funcRetMLIRIt != functionReturnMLIRTypes.end()) {
      resultTy = funcRetMLIRIt->second;
      // Set resultType string if we have it
      auto funcRetTypeIt = functionReturnTypes.find(funcName);
      if (funcRetTypeIt != functionReturnTypes.end()) {
        resultType = funcRetTypeIt->second;
      } else {
        resultType = TypeNames::UNKNOWN;
      }
    } else {
      // Fallback to string-based lookup
      auto funcRetTypeIt = functionReturnTypes.find(funcName);
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
        builder.create<CallOp>(loc(), funcName, TypeRange{resultTy}, args);
    result = callOp.getResult();
  }

  void visit(const NBinaryOperator& node) override {
    node.lhs->accept(*this);
    if (!result) {
      return;
    }
    Value lhs = result;
    std::string lhsType = resultType;

    node.rhs->accept(*this);
    if (!result) {
      return;
    }
    Value rhs = result;

    // Use switch for cleaner operator dispatch
    Type arithResultTy = lhs.getType();
    switch (node.op) {
    // Arithmetic operations - use LHS type as result type
    case yy::parser::token::TPLUS:
      result = builder.create<AddOp>(loc(), arithResultTy, lhs, rhs);
      resultType = lhsType;
      break;
    case yy::parser::token::TMINUS:
      result = builder.create<SubOp>(loc(), arithResultTy, lhs, rhs);
      resultType = lhsType;
      break;
    case yy::parser::token::TMUL:
      result = builder.create<MulOp>(loc(), arithResultTy, lhs, rhs);
      resultType = lhsType;
      break;
    case yy::parser::token::TDIV:
      result = builder.create<DivOp>(loc(), arithResultTy, lhs, rhs);
      resultType = lhsType;
      break;
    // Comparison operations - result is always bool
    case yy::parser::token::TCEQ:
      result = builder.create<CmpOp>(loc(), CmpPredicate::eq, lhs, rhs);
      resultType = TypeNames::BOOL;
      break;
    case yy::parser::token::TCNE:
      result = builder.create<CmpOp>(loc(), CmpPredicate::ne, lhs, rhs);
      resultType = TypeNames::BOOL;
      break;
    case yy::parser::token::TCLT:
      result = builder.create<CmpOp>(loc(), CmpPredicate::lt, lhs, rhs);
      resultType = TypeNames::BOOL;
      break;
    case yy::parser::token::TCLE:
      result = builder.create<CmpOp>(loc(), CmpPredicate::le, lhs, rhs);
      resultType = TypeNames::BOOL;
      break;
    case yy::parser::token::TCGT:
      result = builder.create<CmpOp>(loc(), CmpPredicate::gt, lhs, rhs);
      resultType = TypeNames::BOOL;
      break;
    case yy::parser::token::TCGE:
      result = builder.create<CmpOp>(loc(), CmpPredicate::ge, lhs, rhs);
      resultType = TypeNames::BOOL;
      break;
    default:
      emitError(loc()) << "Unknown binary operator: " << node.op;
      result = nullptr;
      break;
    }
  }

  void visit(const NCastExpression& node) override {
    // Evaluate the expression to cast
    node.expression->accept(*this);
    if (!result) {
      return;
    }
    const Value inputValue = result;

    // Get the target type
    Type targetType = getPolangType(node.targetType->name);

    // Create the cast operation
    result = builder.create<CastOp>(loc(), targetType, inputValue);
    resultType = node.targetType->name;
  }

  void visit(const NAssignment& node) override {
    // Evaluate RHS
    node.rhs->accept(*this);
    if (!result) {
      return;
    }
    Value value = result;
    std::string valueType = resultType;

    // Look up the mutable reference for the variable
    auto refValue = lookupMutableRef(node.lhs->name);
    Value refForStore;
    std::string varType;

    if (refValue) {
      // Found mutable reference
      refForStore = *refValue;
      varType = lookupType(node.lhs->name).value_or(TypeNames::I64);
    } else {
      // Check if it's an immutable value - create immutable ref for verifier
      auto immIt = immutableValues.find(node.lhs->name);
      if (immIt != immutableValues.end()) {
        Value immVal = immIt->second;
        varType = lookupType(node.lhs->name).value_or(TypeNames::I64);
        Type elemType = getPolangType(varType);
        // Create immutable reference - RefStoreOp verifier will catch this
        Type immRefType =
            RefType::get(builder.getContext(), elemType, /*isMutable=*/false);
        auto immRef = builder.create<RefCreateOp>(loc(), immRefType, immVal,
                                                  /*is_mutable=*/false);
        refForStore = immRef.getResult();
      } else {
        emitError(loc()) << "Unknown variable in assignment: "
                         << node.lhs->name;
        result = nullptr;
        return;
      }
    }

    // Get the element type
    Type elemType = getPolangType(getReferentType(varType));

    // Store through the reference (MLIR verifier will catch immutable stores)
    auto storeOp =
        builder.create<RefStoreOp>(loc(), elemType, value, refForStore);

    // Assignment expression returns the assigned value
    result = storeOp.getResult();
    resultType = valueType;
  }

  void visit(const NRefExpression& node) override {
    // Check for `ref *x` pattern where x is a mutable reference
    // In this case, we skip the dereference and directly convert the mutable
    // reference to an immutable reference, preserving the original storage.
    if (const auto* derefExpr =
            dynamic_cast<const NDerefExpression*>(node.expr.get())) {
      derefExpr->ref->accept(*this);
      if (isMutableRefType(resultType)) {
        // Skip the dereference - directly convert mut ref to immutable ref
        Value mutRef = result;
        std::string elemTypeName = getReferentType(resultType);
        Type elemType = getPolangType(elemTypeName);
        Type refType =
            RefType::get(builder.getContext(), elemType, /*isMutable=*/false);
        result = builder.create<RefCreateOp>(loc(), refType, mutRef);
        resultType = makeImmutableRefType(elemTypeName);
        return;
      }
    }

    // Evaluate the inner expression
    node.expr->accept(*this);
    if (!result) {
      return;
    }

    // The inner expression should produce a value
    // We wrap it in an immutable reference type
    Value innerValue = result;
    std::string innerType = resultType;

    // If the inner type is a mutable reference, create an immutable ref from it
    if (isMutableRefType(innerType)) {
      // Convert mut ref to immutable ref
      Type elemType = getPolangType(getReferentType(innerType));
      Type refType =
          RefType::get(builder.getContext(), elemType, /*isMutable=*/false);
      result = builder.create<RefCreateOp>(loc(), refType, innerValue);
      resultType = makeImmutableRefType(getReferentType(innerType));
    } else {
      // The inner value is a plain value; create an immutable reference to it
      // First create a mutable reference to the value, then convert to
      // immutable
      Type innerMLIRType = getPolangType(innerType);

      // Cast the value to the resolved type if needed (e.g., TypeVarType to
      // i32)
      if (innerValue.getType() != innerMLIRType) {
        innerValue = builder.create<CastOp>(loc(), innerMLIRType, innerValue);
      }

      // Create a mutable reference from the value
      Type mutRefType =
          RefType::get(builder.getContext(), innerMLIRType, /*isMutable=*/true);
      auto mutRef = builder.create<RefCreateOp>(loc(), mutRefType, innerValue,
                                                /*is_mutable=*/true);

      // Convert the mutable reference to an immutable reference
      Type refType = RefType::get(builder.getContext(), innerMLIRType,
                                  /*isMutable=*/false);
      result = builder.create<RefCreateOp>(loc(), refType, mutRef.getResult());
      resultType = makeImmutableRefType(innerType);
    }
  }

  void visit(const NDerefExpression& node) override {
    // Evaluate the reference expression
    node.ref->accept(*this);
    if (!result) {
      return;
    }
    Value refValue = result;
    std::string refType = resultType;

    // Check if it's a reference type (mutable or immutable)
    if (isMutableRefType(refType) || isImmutableRefType(refType)) {
      std::string elemTypeName = getReferentType(refType);
      Type elemType = getPolangType(elemTypeName);
      result = builder.create<RefDerefOp>(loc(), elemType, refValue);
      resultType = elemTypeName;
    } else {
      emitError(loc()) << "Cannot dereference non-reference type: " << refType;
      result = nullptr;
    }
  }

  void visit(const NMutRefExpression& node) override {
    // Evaluate the inner expression and pass through the value
    // The MutRefCreateOp is created by NVariableDeclaration when needed
    // NMutRefExpression is purely for AST representation
    node.expr->accept(*this);
    // result and resultType are already set by visiting the inner expression
  }

  void visit(const NBlock& node) override {
    Value lastValue = nullptr;

    for (const auto& stmt : node.statements) {
      stmt->accept(*this);
      if (result) {
        lastValue = result;
      }
    }

    result = lastValue;
  }

  void visit(const NIfExpression& node) override {
    // Evaluate condition
    node.condition->accept(*this);
    if (!result) {
      return;
    }
    Value condition = result;

    // Infer result type from the type checker
    node.thenExpr->accept(*this);
    std::string ifResultType = resultType;

    // Create if operation
    Type resultTy = getPolangType(ifResultType);
    auto ifOp = builder.create<IfOp>(loc(), resultTy, condition);

    // Generate then region
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      node.thenExpr->accept(*this);
      if (result) {
        builder.create<YieldOp>(loc(), result);
      }
    }

    // Generate else region
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      node.elseExpr->accept(*this);
      if (result) {
        builder.create<YieldOp>(loc(), result);
      }
    }

    result = ifOp.getResult();
    resultType = ifResultType;
  }

  void visit(const NLetExpression& node) override {
    // RAII scope guard automatically saves/restores symbol tables
    SymbolTableScope scope(*this);

    // Process bindings
    for (const auto& binding : node.bindings) {
      if (binding->isFunction) {
        binding->func->accept(*this);
      } else {
        binding->var->accept(*this);
      }
    }

    // Evaluate body
    node.body->accept(*this);
  }

  void visit(const NExpressionStatement& node) override {
    node.expression->accept(*this);
  }

  void visit(const NVariableDeclaration& node) override {
    // Get mangled variable name (includes module path)
    const std::string varName = mangledName(node.id->name);

    // Determine the type - prefer type annotation from type checker
    std::string typeName =
        node.type != nullptr ? node.type->name : TypeNames::I64;
    Value initValue = nullptr;
    if (node.assignmentExpr != nullptr) {
      // Evaluate the assignment expression
      node.assignmentExpr->accept(*this);
      initValue = result;

      // Only use resultType if we don't have a type annotation
      // The type checker sets node.type with resolved types
      if (node.type == nullptr) {
        typeName = resultType;
      }
    }

    if (node.isMutable) {
      // Get the base type (strip mut prefix if present for type storage)
      std::string baseTypeName = typeName;
      if (isMutableRefType(baseTypeName)) {
        baseTypeName = getReferentType(baseTypeName);
      }

      // Mutable: Create a mutable reference using the resolved type from type
      // checker
      Type polangType = getPolangType(baseTypeName);
      Type mutRefType =
          RefType::get(builder.getContext(), polangType, /*isMutable=*/true);

      if (initValue) {
        // If initializing from an existing mutable reference, use it directly
        if (auto refType = mlir::dyn_cast<RefType>(initValue.getType())) {
          if (refType.isMutable()) {
            // Value is already a mutable reference (e.g., from
            // NMutRefExpression)
            mutableRefTable[varName] = initValue;
          } else {
            // Create new mutable reference with initial value
            auto mutRefOp =
                builder.create<RefCreateOp>(loc(), mutRefType, initValue,
                                            /*is_mutable=*/true);
            mutableRefTable[varName] = mutRefOp.getResult();
          }
        } else {
          // Create new mutable reference with initial value
          auto mutRefOp =
              builder.create<RefCreateOp>(loc(), mutRefType, initValue,
                                          /*is_mutable=*/true);
          mutableRefTable[varName] = mutRefOp.getResult();
        }
      } else {
        // Create uninitialized mutable reference - use default value
        Type llvmType = convertPolangType(polangType);
        auto memRefType = MemRefType::get({}, llvmType);
        auto alloca = builder.create<AllocaOp>(loc(), memRefType, varName,
                                               polangType, node.isMutable);
        mutableRefTable[varName] = alloca;
      }

      // Store the mutable reference type
      typeTable[varName] = makeMutableRefType(baseTypeName);
    } else {
      // Immutable binding - store to appropriate table based on type
      // NOLINTNEXTLINE(bugprone-branch-clone) - branches store to different
      // maps
      if (isMutableRefType(typeName)) {
        // Copying a mutable reference to an immutable binding
        // Store in mutableRefTable so assignment through the reference works
        mutableRefTable[varName] = initValue;
      } else {
        // Regular immutable value - store SSA value directly (no alloca needed)
        immutableValues[varName] = initValue;
      }
      typeTable[varName] = typeName;
    }

    // For mutable declarations, clear result since the type (mut T) doesn't
    // match what most callers expect (T). For immutable declarations, keep the
    // value.
    if (node.isMutable) {
      result = nullptr;
    } else {
      // For immutable reference types, dereference to match type checker
      // behavior (type checker strips reference types from inferredType for
      // declarations)
      if (isImmutableRefType(typeName)) {
        std::string elemTypeName = getReferentType(typeName);
        Type elemType = getPolangType(elemTypeName);
        result = builder.create<RefDerefOp>(loc(), elemType, initValue);
        resultType = elemTypeName;
      } else {
        result = initValue;
        resultType = typeName;
      }
    }
  }

  void visit(const NFunctionDeclaration& node) override {
    // Save the current insertion point
    OpBuilder::InsertionGuard guard(builder);

    // Get mangled function name (includes module path)
    const std::string funcName = mangledName(node.id->name);

    // Build function type with type variables for untyped parameters
    SmallVector<Type> argTypes;
    std::vector<std::string> argNames;
    std::vector<Type> argMLIRTypes; // Track MLIR types including type vars

    for (const auto& arg : node.arguments) {
      Type argType = getTypeOrFresh(arg->type.get());
      argTypes.push_back(argType);
      argMLIRTypes.push_back(argType);
      argNames.push_back(arg->id->name);
    }

    // Add captured variables as extra parameters
    std::vector<std::string> captureNames;
    std::vector<Type> captureMLIRTypes;
    for (const auto& capture : node.captures) {
      Type captureType = getTypeOrFresh(capture.type.get());
      argTypes.push_back(captureType);
      captureMLIRTypes.push_back(captureType);
      captureNames.push_back(capture.id->name);
    }

    // Store captures for call site (using mangled name)
    functionCaptures[funcName] = captureNames;

    // Return type - use type variable if not specified
    Type returnType = getTypeOrFresh(node.type.get());
    if (node.type != nullptr) {
      functionReturnTypes[funcName] = node.type->name;
    }
    functionReturnMLIRTypes[funcName] = returnType;

    auto funcType = builder.getFunctionType(argTypes, {returnType});

    // Create function at module level
    builder.setInsertionPointToEnd(module.getBody());

    // Convert captureNames to ArrayRef<StringRef>
    SmallVector<StringRef> captureRefs;
    for (const auto& name : captureNames) {
      captureRefs.push_back(name);
    }

    auto funcOp = builder.create<FuncOp>(loc(), funcName, funcType,
                                         ArrayRef<StringRef>(captureRefs));

    // Create entry block with arguments
    Block* entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // RAII scope guard clears and restores symbol tables for function body
    SymbolTableScope scope(*this, /*clearAllTables=*/true);

    // Register function arguments with their MLIR types
    size_t argIdx = 0;
    for (size_t i = 0; i < node.arguments.size(); ++i) {
      const auto& arg = node.arguments[i];
      argValues[arg->id->name] = entryBlock->getArgument(argIdx);
      typeVarTable[arg->id->name] = argMLIRTypes[i];
      if (arg->type != nullptr) {
        typeTable[arg->id->name] = arg->type->name;
      }
      ++argIdx;
    }

    // Register captured variables as arguments
    for (size_t i = 0; i < node.captures.size(); ++i) {
      const auto& capture = node.captures[i];
      argValues[capture.id->name] = entryBlock->getArgument(argIdx);
      typeVarTable[capture.id->name] = captureMLIRTypes[i];
      if (capture.type != nullptr) {
        typeTable[capture.id->name] = capture.type->name;
      }
      ++argIdx;
    }

    // Generate function body
    node.block->accept(*this);

    // Add return - NOLINTNEXTLINE(bugprone-branch-clone) - different op
    // signatures
    if (result) {
      builder.create<ReturnOp>(loc(), result);
    } else {
      builder.create<ReturnOp>(loc());
    }

    result = nullptr; // Function declarations don't produce a value
  }

  void visit(const NModuleDeclaration& node) override {
    // Push module name onto path for name mangling
    currentModulePath.push_back(node.name->name);

    // Generate module members with mangled names
    for (const auto& member : node.members) {
      member->accept(*this);
    }

    // Pop module name from path
    currentModulePath.pop_back();
    result = nullptr; // Module declarations don't produce a value
  }

  void visit(const NImportStatement& node) override {
    // Import statements are processed by the type checker.
    // For MLIR generation, we need to create aliases for imported symbols.
    const std::string moduleName = node.modulePath->mangledName();

    switch (node.kind) {
    case ImportKind::Module:
    case ImportKind::ModuleAlias:
      // Module-level imports don't require MLIR generation.
      // The qualified name access handles the resolution.
      break;

    case ImportKind::Items:
      // from Math import add, PI - create local aliases
      for (const auto& item : node.items) {
        const std::string mangled = moduleName + "$$" + item.name;
        const std::string localName = item.effectiveName();

        // Track the import mapping for name resolution
        importedSymbols[localName] = mangled;

        // Copy variable from mangled name to local name
        auto value = lookupVariable(mangled);
        if (value) {
          immutableValues[localName] = *value;
          auto type = lookupType(mangled);
          if (type) {
            typeTable[localName] = *type;
          }
        }

        // Copy function info from mangled name to local name
        auto retIt = functionReturnTypes.find(mangled);
        if (retIt != functionReturnTypes.end()) {
          functionReturnTypes[localName] = retIt->second;
        }
        auto retMLIRIt = functionReturnMLIRTypes.find(mangled);
        if (retMLIRIt != functionReturnMLIRTypes.end()) {
          functionReturnMLIRTypes[localName] = retMLIRIt->second;
        }
        auto captIt = functionCaptures.find(mangled);
        if (captIt != functionCaptures.end()) {
          functionCaptures[localName] = captIt->second;
        }
      }
      break;

    case ImportKind::All: {
      // from Math import * - import all symbols with the module prefix
      const std::string prefix = moduleName + "$$";

      // Scan function return types for symbols with this prefix
      for (const auto& [mangled, returnType] : functionReturnTypes) {
        if (mangled.substr(0, prefix.size()) == prefix) {
          // Extract local name (after the prefix)
          const std::string localName = mangled.substr(prefix.size());
          // Only import if it's a direct child (no more $$ in the name)
          if (localName.find("$$") == std::string::npos) {
            importedSymbols[localName] = mangled;
          }
        }
      }

      // Scan immutable values for symbols with this prefix
      for (const auto& [mangled, value] : immutableValues) {
        if (mangled.substr(0, prefix.size()) == prefix) {
          const std::string localName = mangled.substr(prefix.size());
          if (localName.find("$$") == std::string::npos) {
            importedSymbols[localName] = mangled;
          }
        }
      }
      break;
    }
    }

    result = nullptr; // Import statements don't produce a value
  }

private:
  OpBuilder builder;
  ModuleOp module;
  TypeChecker typeChecker;

  // Type variable counter for generating fresh type variables
  uint64_t nextTypeVarId = 0;

  // Current result value and type
  Value result;
  std::string resultType;

  // Module path for name mangling (e.g., ["Math", "Internal"])
  std::vector<std::string> currentModulePath;

  // Get mangled name for a symbol within current module context
  [[nodiscard]] std::string mangledName(const std::string& name) const {
    if (currentModulePath.empty()) {
      return name;
    }
    std::string result;
    for (const auto& part : currentModulePath) {
      result += part + "$$";
    }
    result += name;
    return result;
  }

  // Symbol tables
  std::map<std::string, Value> mutableRefTable; // Mutable reference values
  std::map<std::string, Value> argValues;       // Function arguments
  std::map<std::string, Value> immutableValues; // Immutable variable SSA values
  std::map<std::string, std::string> typeTable; // Variable types
  std::map<std::string, Type>
      typeVarTable; // Variable types as MLIR types (for type vars)
  std::map<std::string, std::vector<std::string>> functionCaptures;
  std::map<std::string, std::string> functionReturnTypes;
  std::map<std::string, Type>
      functionReturnMLIRTypes; // Function return types as MLIR types
  std::map<std::string, std::string>
      importedSymbols; // local name -> mangled name

  /// RAII helper class for scoped symbol table management.
  /// Automatically saves and restores symbol tables when entering/exiting
  /// scopes.
  class SymbolTableScope {
  public:
    SymbolTableScope(MLIRGenVisitor& visitor, bool clearAllTables = false)
        : visitor(visitor), savedMutableRefTable(visitor.mutableRefTable),
          savedTypeTable(visitor.typeTable),
          savedTypeVarTable(visitor.typeVarTable),
          savedArgValues(visitor.argValues),
          savedImmutableValues(visitor.immutableValues) {
      if (clearAllTables) {
        visitor.mutableRefTable.clear();
        visitor.typeTable.clear();
        visitor.typeVarTable.clear();
        visitor.argValues.clear();
        visitor.immutableValues.clear();
      }
    }

    // NOLINTNEXTLINE(modernize-use-equals-default) - destructor restores saved
    // state
    ~SymbolTableScope() {
      visitor.mutableRefTable = savedMutableRefTable;
      visitor.typeTable = savedTypeTable;
      visitor.typeVarTable = savedTypeVarTable;
      visitor.argValues = savedArgValues;
      visitor.immutableValues = savedImmutableValues;
    }

  private:
    MLIRGenVisitor& visitor;
    std::map<std::string, Value> savedMutableRefTable;
    std::map<std::string, std::string> savedTypeTable;
    std::map<std::string, Type> savedTypeVarTable;
    std::map<std::string, Value> savedArgValues;
    std::map<std::string, Value> savedImmutableValues;
  };

  Location loc() { return builder.getUnknownLoc(); }

  /// Look up a mutable reference by name.
  /// Returns the reference value itself (not dereferenced).
  std::optional<Value> lookupMutableRef(const std::string& name) {
    auto refIt = mutableRefTable.find(name);
    if (refIt != mutableRefTable.end()) {
      return refIt->second;
    }
    return std::nullopt;
  }

  /// Look up a variable by name in the symbol tables.
  /// For mutable references, returns the reference (not dereferenced).
  /// Checks immutable values, mutable references, and function arguments.
  /// Returns nullopt if the variable is not found.
  std::optional<Value> lookupVariable(const std::string& name) {
    // Check immutable values first (SSA values, no load needed)
    auto immIt = immutableValues.find(name);
    if (immIt != immutableValues.end()) {
      return immIt->second;
    }

    // Check mutable references - return the reference itself
    auto refIt = mutableRefTable.find(name);
    if (refIt != mutableRefTable.end()) {
      return refIt->second;
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
    // Handle reference types
    if (isMutableRefType(typeName)) {
      std::string elemTypeName = getReferentType(typeName);
      Type elemType = getPolangType(elemTypeName);
      return RefType::get(builder.getContext(), elemType, /*isMutable=*/true);
    }
    if (isImmutableRefType(typeName)) {
      std::string elemTypeName = getReferentType(typeName);
      Type elemType = getPolangType(elemTypeName);
      return RefType::get(builder.getContext(), elemType, /*isMutable=*/false);
    }
    // Signed integers
    if (typeName == TypeNames::I8) {
      return polang::IntegerType::get(builder.getContext(), 8,
                                      Signedness::Signed);
    }
    if (typeName == TypeNames::I16) {
      return polang::IntegerType::get(builder.getContext(), 16,
                                      Signedness::Signed);
    }
    if (typeName == TypeNames::I32) {
      return polang::IntegerType::get(builder.getContext(), 32,
                                      Signedness::Signed);
    }
    if (typeName == TypeNames::I64) {
      return polang::IntegerType::get(builder.getContext(), 64,
                                      Signedness::Signed);
    }
    // Unsigned integers
    if (typeName == TypeNames::U8) {
      return polang::IntegerType::get(builder.getContext(), 8,
                                      Signedness::Unsigned);
    }
    if (typeName == TypeNames::U16) {
      return polang::IntegerType::get(builder.getContext(), 16,
                                      Signedness::Unsigned);
    }
    if (typeName == TypeNames::U32) {
      return polang::IntegerType::get(builder.getContext(), 32,
                                      Signedness::Unsigned);
    }
    if (typeName == TypeNames::U64) {
      return polang::IntegerType::get(builder.getContext(), 64,
                                      Signedness::Unsigned);
    }
    // Floats
    if (typeName == TypeNames::F32) {
      return polang::FloatType::get(builder.getContext(), 32);
    }
    if (typeName == TypeNames::F64) {
      return polang::FloatType::get(builder.getContext(), 64);
    }
    // Bool
    if (typeName == TypeNames::BOOL) {
      return builder.getType<BoolType>();
    }
    // Type variable
    if (typeName == TypeNames::TYPEVAR) {
      return freshTypeVar();
    }
    // Generic types (unresolved literals)
    if (typeName == TypeNames::GENERIC_INT) {
      return freshTypeVar(TypeVarKind::Integer);
    }
    if (typeName == TypeNames::GENERIC_FLOAT) {
      return freshTypeVar(TypeVarKind::Float);
    }
    // Default to i64
    return polang::IntegerType::get(builder.getContext(), 64,
                                    Signedness::Signed);
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
    return polang::IntegerType::get(builder.getContext(), 64,
                                    Signedness::Signed);
  }

  Type convertPolangType(Type polangType) {
    if (auto intType = dyn_cast<polang::IntegerType>(polangType)) {
      return builder.getIntegerType(intType.getWidth());
    }
    if (auto floatType = dyn_cast<polang::FloatType>(polangType)) {
      if (floatType.getWidth() == 32) {
        return builder.getF32Type();
      }
      return builder.getF64Type();
    }
    if (isa<BoolType>(polangType)) {
      return builder.getI1Type();
    }
    return builder.getI64Type();
  }

  void generateMainFunction(const NBlock& block) {
    // Get the inferred return type from the type checker
    std::string inferredType = typeChecker.getInferredType();

    // Use type checker's type if it's concrete, otherwise use type variable
    Type returnType;
    if (inferredType == TypeNames::UNKNOWN ||
        inferredType == TypeNames::TYPEVAR || isGenericType(inferredType)) {
      // Type is unknown or generic - use type variable for MLIR inference
      returnType = freshTypeVar();
    } else {
      // Type checker found a concrete type - use it
      returnType = getPolangType(inferredType);
    }

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
      if (isFloatType(inferredType)) {
        auto floatTy = polang::FloatType::get(builder.getContext(),
                                              getFloatWidth(inferredType));
        defaultVal = builder.create<ConstantFloatOp>(loc(), floatTy, 0.0);
      } else if (inferredType == TypeNames::BOOL) {
        // NOLINTNEXTLINE(bugprone-branch-clone) - creates different op types
        defaultVal = builder.create<ConstantBoolOp>(loc(), false);
      } else {
        // Use the return type (already resolved or type variable)
        defaultVal = builder.create<ConstantIntegerOp>(loc(), returnType, 0);
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
  if (!module) {
    return nullptr;
  }

  return module;
}
