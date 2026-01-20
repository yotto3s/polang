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

/// Check if a variable declaration is mutable based on:
/// 1. Type annotation has "mut " prefix, OR
/// 2. Assignment expression is NMutRefExpression (for inferred types)
[[nodiscard]] bool isVarDeclMutable(const NVariableDeclaration& decl) {
  // Check type annotation first
  if (decl.type != nullptr && isMutableRefType(decl.type->getTypeName())) {
    return true;
  }
  // Check if assignment expression is NMutRefExpression
  if (decl.assignmentExpr != nullptr) {
    if (dynamic_cast<const NMutRefExpression*>(decl.assignmentExpr.get()) !=
        nullptr) {
      return true;
    }
  }
  return false;
}

/// Helper to create an NTypeSpec from a type name string.
/// Handles reference types (e.g., "mut i64", "ref i32") recursively.
[[nodiscard]] std::shared_ptr<const NTypeSpec>
makeTypeSpecFromString(const std::string& typeName) {
  if (isMutableRefType(typeName)) {
    return std::make_shared<const NMutRefType>(
        makeTypeSpecFromString(getReferentType(typeName)));
  }
  if (isImmutableRefType(typeName)) {
    return std::make_shared<const NRefType>(
        makeTypeSpecFromString(getReferentType(typeName)));
  }
  return std::make_shared<const NNamedType>(typeName);
}

//===----------------------------------------------------------------------===//
// MLIRGenVisitor - Generates MLIR from Polang AST
//===----------------------------------------------------------------------===//

class MLIRGenVisitor : public Visitor {
public:
  MLIRGenVisitor(MLIRContext& context, bool /*emitTypeVars*/ = false,
                 const std::string& filename = "<source>")
      : builder(&context), sourceFilename(filename) {
    // Create a new module
    module = ModuleOp::create(builder.getUnknownLoc());
  }

  /// Generate a fresh type variable with optional kind constraint
  Type freshTypeVar(TypeVarKind kind = TypeVarKind::Any) {
    return builder.getType<TypeVarType>(nextTypeVarId++, kind);
  }

  /// Get a Polang type from annotation, or a fresh type variable if none
  Type getTypeOrFresh(const NTypeSpec* typeAnnotation) {
    if (typeAnnotation != nullptr) {
      Type ty = getPolangType(*typeAnnotation);
      if (!ty) {
        return nullptr; // Propagate error
      }
      return ty;
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

    // Check if any errors occurred during MLIR generation
    if (hasMLIRGenErrors) {
      return nullptr;
    }

    // Skip module verification since type variables may be present.
    // Type inference pass will resolve them and verification will happen later.
    // We keep verification disabled to allow polymorphic types.
    // Note: Verification is intentionally disabled here.

    return module;
  }

  // Type Specification Visitor implementations
  void visit(const NNamedType& /*node*/) override {
    // Type specifications are not directly visited during MLIR generation
    // They are accessed via getTypeName() when processing declarations
  }

  void visit(const NRefType& /*node*/) override {
    // Type specifications are not directly visited during MLIR generation
  }

  void visit(const NMutRefType& /*node*/) override {
    // Type specifications are not directly visited during MLIR generation
  }

  // Expression Visitor implementations
  void visit(const NInteger& node) override {
    // Create type variable constrained to integer types
    auto type = freshTypeVar(TypeVarKind::Integer);
    result = builder.create<ConstantIntegerOp>(loc(node.loc), type, node.value);
    resultType = std::make_shared<const NNamedType>(TypeNames::GENERIC_INT);
  }

  void visit(const NDouble& node) override {
    // Create type variable constrained to float types
    auto type = freshTypeVar(TypeVarKind::Float);
    result = builder.create<ConstantFloatOp>(loc(node.loc), type, node.value);
    resultType = std::make_shared<const NNamedType>(TypeNames::GENERIC_FLOAT);
  }

  void visit(const NBoolean& node) override {
    result = builder.create<ConstantBoolOp>(loc(node.loc), node.value);
    resultType = std::make_shared<const NNamedType>(TypeNames::BOOL);
  }

  void visit(const NIdentifier& node) override {
    auto value = lookupVariable(node.name);
    if (value) {
      result = *value;
      auto type = lookupType(node.name);
      resultType =
          type ? type : std::make_shared<const NNamedType>(TypeNames::I64);
      return;
    }

    emitError(loc(node.loc)) << "Unknown variable: " << node.name;
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
      resultType =
          type ? type : std::make_shared<const NNamedType>(TypeNames::I64);
      return;
    }

    emitError(loc(node.loc)) << "Unknown qualified name: " << node.fullName();
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
      // Set resultType if we have it
      auto funcRetTypeIt = functionReturnTypes.find(funcName);
      if (funcRetTypeIt != functionReturnTypes.end()) {
        resultType = funcRetTypeIt->second;
      } else {
        resultType = std::make_shared<const NNamedType>(TypeNames::UNKNOWN);
      }
    } else {
      // Fallback to NTypeSpec-based lookup
      auto funcRetTypeIt = functionReturnTypes.find(funcName);
      if (funcRetTypeIt != functionReturnTypes.end()) {
        resultTy = getPolangType(*funcRetTypeIt->second);
        if (!resultTy) {
          result = nullptr;
          return;
        }
        resultType = funcRetTypeIt->second;
      } else {
        // Function not found - generate fresh type var for unknown function
        resultTy = freshTypeVar();
        resultType = std::make_shared<const NNamedType>(TypeNames::UNKNOWN);
      }
    }

    auto callOp = builder.create<CallOp>(loc(node.loc), funcName,
                                         TypeRange{resultTy}, args);
    result = callOp.getResult();
  }

  void visit(const NBinaryOperator& node) override {
    node.lhs->accept(*this);
    if (!result) {
      return;
    }
    Value lhs = result;
    std::shared_ptr<const NTypeSpec> lhsType = std::move(resultType);

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
      result = builder.create<AddOp>(loc(node.loc), arithResultTy, lhs, rhs);
      resultType = std::move(lhsType);
      break;
    case yy::parser::token::TMINUS:
      result = builder.create<SubOp>(loc(node.loc), arithResultTy, lhs, rhs);
      resultType = std::move(lhsType);
      break;
    case yy::parser::token::TMUL:
      result = builder.create<MulOp>(loc(node.loc), arithResultTy, lhs, rhs);
      resultType = std::move(lhsType);
      break;
    case yy::parser::token::TDIV:
      result = builder.create<DivOp>(loc(node.loc), arithResultTy, lhs, rhs);
      resultType = std::move(lhsType);
      break;
    // Comparison operations - result is always bool
    case yy::parser::token::TCEQ:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::eq, lhs, rhs);
      resultType = std::make_shared<const NNamedType>(TypeNames::BOOL);
      break;
    case yy::parser::token::TCNE:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::ne, lhs, rhs);
      resultType = std::make_shared<const NNamedType>(TypeNames::BOOL);
      break;
    case yy::parser::token::TCLT:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::lt, lhs, rhs);
      resultType = std::make_shared<const NNamedType>(TypeNames::BOOL);
      break;
    case yy::parser::token::TCLE:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::le, lhs, rhs);
      resultType = std::make_shared<const NNamedType>(TypeNames::BOOL);
      break;
    case yy::parser::token::TCGT:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::gt, lhs, rhs);
      resultType = std::make_shared<const NNamedType>(TypeNames::BOOL);
      break;
    case yy::parser::token::TCGE:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::ge, lhs, rhs);
      resultType = std::make_shared<const NNamedType>(TypeNames::BOOL);
      break;
    default:
      emitError(loc(node.loc)) << "Unknown binary operator: " << node.op;
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
    Type targetType = getPolangType(*node.targetType);
    if (!targetType) {
      result = nullptr;
      return;
    }

    // Create the cast operation
    result = builder.create<CastOp>(loc(node.loc), targetType, inputValue);
    resultType = node.targetType;
  }

  void visit(const NAssignment& node) override {
    // Evaluate RHS
    node.rhs->accept(*this);
    if (!result) {
      return;
    }
    Value value = result;
    std::shared_ptr<const NTypeSpec> valueType = std::move(resultType);

    // Look up the mutable reference for the variable
    auto refValue = lookupMutableRef(node.lhs->name);
    Value refForStore;
    std::shared_ptr<const NTypeSpec> varType;

    if (refValue) {
      // Found mutable reference
      refForStore = *refValue;
      varType = lookupType(node.lhs->name);
    } else {
      // Check if it's an immutable value - create immutable ref for verifier
      auto immIt = immutableValues.find(node.lhs->name);
      if (immIt != immutableValues.end()) {
        Value immVal = immIt->second;
        varType = lookupType(node.lhs->name);
        Type elemType =
            varType != nullptr ? getPolangType(*varType) : getDefaultType();
        // Create immutable reference - RefStoreOp verifier will catch this
        Type immRefType =
            RefType::get(builder.getContext(), elemType, /*isMutable=*/false);
        auto immRef =
            builder.create<RefCreateOp>(loc(node.loc), immRefType, immVal,
                                        /*is_mutable=*/false);
        refForStore = immRef.getResult();
      } else {
        emitError(loc(node.loc))
            << "Unknown variable in assignment: " << node.lhs->name;
        result = nullptr;
        return;
      }
    }

    // Get the element type from the referent
    Type elemType = getReferentMLIRType(varType.get());

    // Store through the reference (MLIR verifier will catch immutable stores)
    auto storeOp =
        builder.create<RefStoreOp>(loc(node.loc), elemType, value, refForStore);

    // Assignment expression returns the assigned value
    result = storeOp.getResult();
    resultType = std::move(valueType);
  }

  void visit(const NRefExpression& node) override {
    // Check for `ref *x` pattern where x is a mutable reference
    // In this case, we skip the dereference and directly convert the mutable
    // reference to an immutable reference, preserving the original storage.
    if (const auto* derefExpr =
            dynamic_cast<const NDerefExpression*>(node.expr.get())) {
      derefExpr->ref->accept(*this);
      if (isMutRef(resultType.get())) {
        // Skip the dereference - directly convert mut ref to immutable ref
        Value mutRef = result;
        auto innerType = getInnerTypeSpec(resultType.get());
        Type elemType =
            innerType ? getPolangType(*innerType) : getDefaultType();
        Type refType =
            RefType::get(builder.getContext(), elemType, /*isMutable=*/false);
        result = builder.create<RefCreateOp>(loc(node.loc), refType, mutRef);
        resultType =
            innerType ? std::make_shared<const NRefType>(innerType)
                      : std::make_shared<const NRefType>(
                            std::make_shared<const NNamedType>(TypeNames::I64));
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
    std::shared_ptr<const NTypeSpec> innerType = std::move(resultType);

    // If the inner type is a mutable reference, create an immutable ref from it
    if (isMutRef(innerType.get())) {
      // Convert mut ref to immutable ref
      auto inner = getInnerTypeSpec(innerType.get());
      Type elemType = inner ? getPolangType(*inner) : getDefaultType();
      Type refType =
          RefType::get(builder.getContext(), elemType, /*isMutable=*/false);
      result = builder.create<RefCreateOp>(loc(node.loc), refType, innerValue);
      resultType =
          inner ? std::make_shared<const NRefType>(inner)
                : std::make_shared<const NRefType>(
                      std::make_shared<const NNamedType>(TypeNames::I64));
    } else {
      // The inner value is a plain value; create an immutable reference to it
      // First create a mutable reference to the value, then convert to
      // immutable
      Type innerMLIRType =
          innerType ? getPolangType(*innerType) : getDefaultType();

      // Cast the value to the resolved type if needed (e.g., TypeVarType to
      // i32)
      if (innerValue.getType() != innerMLIRType) {
        innerValue =
            builder.create<CastOp>(loc(node.loc), innerMLIRType, innerValue);
      }

      // Create a mutable reference from the value
      Type mutRefType =
          RefType::get(builder.getContext(), innerMLIRType, /*isMutable=*/true);
      auto mutRef =
          builder.create<RefCreateOp>(loc(node.loc), mutRefType, innerValue,
                                      /*is_mutable=*/true);

      // Convert the mutable reference to an immutable reference
      Type refType = RefType::get(builder.getContext(), innerMLIRType,
                                  /*isMutable=*/false);
      result = builder.create<RefCreateOp>(loc(node.loc), refType,
                                           mutRef.getResult());
      resultType =
          innerType ? std::make_shared<const NRefType>(innerType)
                    : std::make_shared<const NRefType>(
                          std::make_shared<const NNamedType>(TypeNames::I64));
    }
  }

  void visit(const NDerefExpression& node) override {
    // Evaluate the reference expression
    node.ref->accept(*this);
    if (!result) {
      return;
    }
    Value refValue = result;

    // Check if it's a reference type (mutable or immutable)
    if (isMutRef(resultType.get()) || isImmutRef(resultType.get())) {
      auto innerTypeSpec = getInnerTypeSpec(resultType.get());
      Type elemType = innerTypeSpec != nullptr ? getPolangType(*innerTypeSpec)
                                               : getDefaultType();
      result = builder.create<RefDerefOp>(loc(node.loc), elemType, refValue);
      resultType = innerTypeSpec != nullptr
                       ? innerTypeSpec
                       : std::make_shared<const NNamedType>(TypeNames::I64);
    } else {
      emitError(loc(node.loc))
          << "Cannot dereference non-reference type: "
          << (resultType ? resultType->getTypeName() : "unknown");
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
    std::shared_ptr<const NTypeSpec> ifResultType = std::move(resultType);

    // Create if operation
    Type resultTy =
        ifResultType ? getPolangType(*ifResultType) : getDefaultType();
    auto ifOp = builder.create<IfOp>(loc(node.loc), resultTy, condition);

    // Generate then region
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      node.thenExpr->accept(*this);
      if (result) {
        builder.create<YieldOp>(loc(node.thenExpr->loc), result);
      }
    }

    // Generate else region
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      node.elseExpr->accept(*this);
      if (result) {
        builder.create<YieldOp>(loc(node.elseExpr->loc), result);
      }
    }

    result = ifOp.getResult();
    resultType = std::move(ifResultType);
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
    // Derive mutability from type annotation or NMutRefExpression
    const bool isMutable = isVarDeclMutable(node);

    // Determine the type - prefer type annotation from type checker
    std::shared_ptr<const NTypeSpec> typeSpec =
        node.type != nullptr
            ? node.type
            : std::make_shared<const NNamedType>(TypeNames::I64);
    Value initValue = nullptr;
    if (node.assignmentExpr != nullptr) {
      // Evaluate the assignment expression
      node.assignmentExpr->accept(*this);
      initValue = result;

      // Only use resultType if we don't have a type annotation
      // The type checker sets node.type with resolved types
      if (node.type == nullptr && resultType) {
        typeSpec = std::move(resultType);
      }
    }

    if (isMutable) {
      // Get the base type (strip mut prefix if present for type storage)
      std::shared_ptr<const NTypeSpec> baseType;
      if (isMutRef(typeSpec.get())) {
        auto inner = getInnerTypeSpec(typeSpec.get());
        baseType = inner != nullptr
                       ? inner
                       : std::make_shared<const NNamedType>(TypeNames::I64);
      } else {
        baseType = typeSpec;
      }

      // Mutable: Create a mutable reference using the resolved type from type
      // checker
      Type polangType = getPolangType(*baseType);
      if (!polangType) {
        result = nullptr;
        return;
      }
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
            auto mutRefOp = builder.create<RefCreateOp>(loc(node.loc),
                                                        mutRefType, initValue,
                                                        /*is_mutable=*/true);
            mutableRefTable[varName] = mutRefOp.getResult();
          }
        } else {
          // Create new mutable reference with initial value
          auto mutRefOp =
              builder.create<RefCreateOp>(loc(node.loc), mutRefType, initValue,
                                          /*is_mutable=*/true);
          mutableRefTable[varName] = mutRefOp.getResult();
        }
      } else {
        // Create uninitialized mutable reference - use default value
        Type llvmType = convertPolangType(polangType);
        auto memRefType = MemRefType::get({}, llvmType);
        auto alloca = builder.create<AllocaOp>(loc(node.loc), memRefType,
                                               varName, polangType, isMutable);
        mutableRefTable[varName] = alloca;
      }

      // Store the mutable reference type
      typeTable[varName] = std::make_shared<const NMutRefType>(baseType);
    } else {
      // Immutable binding - store to appropriate table based on type
      // NOLINTNEXTLINE(bugprone-branch-clone) - branches store to different
      // maps
      if (isMutRef(typeSpec.get())) {
        // Copying a mutable reference to an immutable binding
        // Store in mutableRefTable so assignment through the reference works
        mutableRefTable[varName] = initValue;
      } else {
        // Regular immutable value - store SSA value directly (no alloca needed)
        immutableValues[varName] = initValue;
      }
      typeTable[varName] = typeSpec;
    }

    // For mutable declarations, clear result since the type (mut T) doesn't
    // match what most callers expect (T). For immutable declarations, keep the
    // value.
    if (isMutable) {
      result = nullptr;
    } else {
      // For immutable reference types, dereference to match type checker
      // behavior (type checker strips reference types from inferredType for
      // declarations)
      if (isImmutRef(typeSpec.get())) {
        auto innerType = getInnerTypeSpec(typeSpec.get());
        Type elemType =
            innerType != nullptr ? getPolangType(*innerType) : getDefaultType();
        result = builder.create<RefDerefOp>(loc(node.loc), elemType, initValue);
        resultType = innerType != nullptr
                         ? innerType
                         : std::make_shared<const NNamedType>(TypeNames::I64);
      } else {
        result = initValue;
        resultType = typeSpec;
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
      functionReturnTypes[funcName] = node.type;
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

    auto funcOp = builder.create<FuncOp>(loc(node.loc), funcName, funcType,
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
        typeTable[arg->id->name] = arg->type;
      }
      ++argIdx;
    }

    // Register captured variables as arguments
    for (size_t i = 0; i < node.captures.size(); ++i) {
      const auto& capture = node.captures[i];
      argValues[capture.id->name] = entryBlock->getArgument(argIdx);
      typeVarTable[capture.id->name] = captureMLIRTypes[i];
      if (capture.type != nullptr) {
        typeTable[capture.id->name] = capture.type;
      }
      ++argIdx;
    }

    // Generate function body
    node.block->accept(*this);

    // Add return - NOLINTNEXTLINE(bugprone-branch-clone) - different op
    // signatures
    if (result) {
      builder.create<ReturnOp>(loc(node.block->loc), result);
    } else {
      builder.create<ReturnOp>(loc(node.loc));
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
            typeTable[localName] = type;
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
  std::string sourceFilename;

  // Type variable counter for generating fresh type variables
  uint64_t nextTypeVarId = 0;

  // Track whether any errors occurred during MLIR generation
  bool hasMLIRGenErrors = false;

  // Current result value and type
  Value result;
  std::shared_ptr<const NTypeSpec> resultType;

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
  std::map<std::string, std::shared_ptr<const NTypeSpec>>
      typeTable; // Variable types
  std::map<std::string, Type>
      typeVarTable; // Variable types as MLIR types (for type vars)
  std::map<std::string, std::vector<std::string>> functionCaptures;
  std::map<std::string, std::shared_ptr<const NTypeSpec>> functionReturnTypes;
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
    std::map<std::string, std::shared_ptr<const NTypeSpec>> savedTypeTable;
    std::map<std::string, Type> savedTypeVarTable;
    std::map<std::string, Value> savedArgValues;
    std::map<std::string, Value> savedImmutableValues;
  };

  /// Create MLIR location from source location
  Location loc(const SourceLocation& srcLoc) {
    if (srcLoc.isValid()) {
      return FileLineColLoc::get(builder.getContext(), sourceFilename,
                                 srcLoc.line, srcLoc.column);
    }
    return builder.getUnknownLoc();
  }

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
  /// Returns nullptr if not found.
  [[nodiscard]] std::shared_ptr<const NTypeSpec>
  lookupType(const std::string& name) {
    auto it = typeTable.find(name);
    if (it != typeTable.end()) {
      return it->second;
    }
    return nullptr;
  }

  /// Get the default type (i64) for cases where no type is specified.
  Type getDefaultType() {
    return polang::IntegerType::get(builder.getContext(), 64,
                                    Signedness::Signed);
  }

  /// Get a Polang MLIR type from an NTypeSpec.
  /// Handles NNamedType, NRefType, and NMutRefType.
  /// Rejects nested reference types (e.g., ref ref i64).
  Type getPolangType(const NTypeSpec& typeSpec) {
    // Handle NMutRefType
    if (const auto* mutRef = dynamic_cast<const NMutRefType*>(&typeSpec)) {
      // Reject nested reference types
      if (dynamic_cast<const NRefType*>(mutRef->innerType.get()) != nullptr ||
          dynamic_cast<const NMutRefType*>(mutRef->innerType.get()) !=
              nullptr) {
        emitError(builder.getUnknownLoc())
            << "nested reference types not allowed";
        hasMLIRGenErrors = true;
        return nullptr;
      }
      Type elemType = getPolangType(*mutRef->innerType);
      if (!elemType) {
        return nullptr;
      }
      return RefType::get(builder.getContext(), elemType, /*isMutable=*/true);
    }

    // Handle NRefType
    if (const auto* ref = dynamic_cast<const NRefType*>(&typeSpec)) {
      // Reject nested reference types
      if (dynamic_cast<const NRefType*>(ref->innerType.get()) != nullptr ||
          dynamic_cast<const NMutRefType*>(ref->innerType.get()) != nullptr) {
        emitError(builder.getUnknownLoc())
            << "nested reference types not allowed";
        hasMLIRGenErrors = true;
        return nullptr;
      }
      Type elemType = getPolangType(*ref->innerType);
      if (!elemType) {
        return nullptr;
      }
      return RefType::get(builder.getContext(), elemType, /*isMutable=*/false);
    }

    // Handle NNamedType
    if (const auto* named = dynamic_cast<const NNamedType*>(&typeSpec)) {
      const std::string& typeName = named->name;

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
      return getDefaultType();
    }

    // Unknown type - should not happen
    return nullptr;
  }

  /// Get the MLIR type for the referent of a reference type.
  /// If the type is a reference, returns the inner type.
  /// If not a reference, returns the type itself.
  Type getReferentMLIRType(const NTypeSpec* typeSpec) {
    if (typeSpec == nullptr) {
      return getDefaultType();
    }
    if (const auto* mutRef = dynamic_cast<const NMutRefType*>(typeSpec)) {
      return getPolangType(*mutRef->innerType);
    }
    if (const auto* ref = dynamic_cast<const NRefType*>(typeSpec)) {
      return getPolangType(*ref->innerType);
    }
    return getPolangType(*typeSpec);
  }

  /// Check if an NTypeSpec represents a mutable reference type.
  [[nodiscard]] static bool isMutRef(const NTypeSpec* typeSpec) {
    return dynamic_cast<const NMutRefType*>(typeSpec) != nullptr;
  }

  /// Check if an NTypeSpec represents an immutable reference type.
  [[nodiscard]] static bool isImmutRef(const NTypeSpec* typeSpec) {
    return dynamic_cast<const NRefType*>(typeSpec) != nullptr &&
           dynamic_cast<const NMutRefType*>(typeSpec) == nullptr;
  }

  /// Get the inner type spec from a reference type.
  /// Returns nullptr if not a reference type.
  [[nodiscard]] static std::shared_ptr<const NTypeSpec>
  getInnerTypeSpec(const NTypeSpec* typeSpec) {
    if (const auto* mutRef = dynamic_cast<const NMutRefType*>(typeSpec)) {
      return mutRef->innerType;
    }
    if (const auto* ref = dynamic_cast<const NRefType*>(typeSpec)) {
      return ref->innerType;
    }
    return nullptr;
  }

  Type getTypeForName(const std::string& name) {
    // First check typeVarTable for MLIR types (including type vars)
    auto varIt = typeVarTable.find(name);
    if (varIt != typeVarTable.end()) {
      return varIt->second;
    }
    // Fall back to NTypeSpec-based type table
    auto it = typeTable.find(name);
    if (it != typeTable.end() && it->second) {
      return getPolangType(*it->second);
    }
    return getDefaultType();
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
      auto typeSpec = makeTypeSpecFromString(inferredType);
      returnType = getPolangType(*typeSpec);
      if (!returnType) {
        return; // Stop generating if type error
      }
    }

    // Create entry function with dynamic return type
    auto funcType = builder.getFunctionType({}, {returnType});

    builder.setInsertionPointToEnd(module.getBody());
    auto entryFunc =
        builder.create<FuncOp>(loc(block.loc), "__polang_entry", funcType);

    // Create entry block
    Block* entryBlock = entryFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate code for the block
    block.accept(*this);

    // Return the last expression value, or default value of correct type
    if (result) {
      builder.create<ReturnOp>(loc(block.loc), result);
    } else {
      // Create default value matching the return type
      Value defaultVal;
      if (isFloatType(inferredType)) {
        auto floatTy = polang::FloatType::get(builder.getContext(),
                                              getFloatWidth(inferredType));
        defaultVal =
            builder.create<ConstantFloatOp>(loc(block.loc), floatTy, 0.0);
      } else if (inferredType == TypeNames::BOOL) {
        // NOLINTNEXTLINE(bugprone-branch-clone) - creates different op types
        defaultVal = builder.create<ConstantBoolOp>(loc(block.loc), false);
      } else {
        // Use the return type (already resolved or type variable)
        defaultVal =
            builder.create<ConstantIntegerOp>(loc(block.loc), returnType, 0);
      }
      builder.create<ReturnOp>(loc(block.loc), defaultVal);
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
