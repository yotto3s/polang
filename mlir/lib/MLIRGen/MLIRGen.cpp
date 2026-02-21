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
#include "polang/PolangTypeConverter.h"

// clang-format off
#include "parser/node.hpp"
#include "parser.hpp"  // Must be after node.hpp for bison union types
// clang-format on
#include "parser/polang_types.hpp"
#include "parser/type_inference.hpp"
#include "parser/visitor.hpp"

using polang::TypeNames;

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#pragma GCC diagnostic pop

#include "compiler/compiled_symbols.hpp"

#include <map>
#include <optional>
#include <set>
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
  MLIRGenVisitor(MLIRContext& context, bool /*emitTypeVars*/ = false,
                 std::string inferredType = "", std::string entryFuncName = "",
                 std::string filename = "<source>",
                 OptCompiledSymbols compiledSymbols = std::nullopt)
      : builder(&context), typeConverter(&context),
        inferredType(std::move(inferredType)),
        entryFuncName(entryFuncName.empty() ? "__polang_entry"
                                            : std::move(entryFuncName)),
        sourceFilename(std::move(filename)), compiledSymbols(compiledSymbols) {
    // Create a new module
    module = ModuleOp::create(builder.getUnknownLoc());
  }

  /// Get a fresh type variable (when no type annotation is present)
  Type getTypeFresh() { return typeConverter.getTypeOrFresh(nullptr); }

  /// Get a Polang type from a type annotation
  Type getType(const NTypeSpec& typeAnnotation) {
    return typeConverter.getPolangType(typeAnnotation);
  }

  /// Generate MLIR for the given AST block
  ModuleOp generate(const NBlock& block) {
    // In incremental mode, populate extern declarations from previous evals
    if (isIncremental()) {
      populateFromCompiledSymbols();
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
  void visit(const NArrowType& /*node*/) override {}
  void visit(const NProductType& /*node*/) override {}
  void visit(const NTypeVar& /*node*/) override {}
  void visit(const NForallType& /*node*/) override {}
  void visit(const NUnitType& /*node*/) override {}

  // Expression Visitor implementations
  void visit(const NInteger& node) override {
    // If there's an expected integer type from a variable declaration
    // annotation, use it instead of the default i64.
    if (expectedLiteralType != nullptr) {
      const auto meta = getTypeMetadata(expectedLiteralType->getTypeName());
      if (meta.isInteger() || meta.isIndex()) {
        auto type = getPolangType(*expectedLiteralType);
        result =
            builder.create<ConstantIntegerOp>(loc(node.loc), type, node.value);
        resultType = expectedLiteralType->clone();
        return;
      }
    }
    auto type = getDefaultType();
    result = builder.create<ConstantIntegerOp>(loc(node.loc), type, node.value);
    resultType = makeTypeSpec(TypeNames::GENERIC_INT);
  }

  void visit(const NDouble& node) override {
    // If there's an expected float type from a variable declaration annotation,
    // use it instead of the default f64.
    if (expectedLiteralType != nullptr) {
      const auto meta = getTypeMetadata(expectedLiteralType->getTypeName());
      if (meta.isFloat()) {
        auto type = polang::FloatType::get(builder.getContext(), meta.width);
        result =
            builder.create<ConstantFloatOp>(loc(node.loc), type, node.value);
        resultType = expectedLiteralType->clone();
        return;
      }
    }
    auto type =
        polang::FloatType::get(builder.getContext(), DEFAULT_FLOAT_WIDTH);
    result = builder.create<ConstantFloatOp>(loc(node.loc), type, node.value);
    resultType = makeTypeSpec(TypeNames::GENERIC_FLOAT);
  }

  void visit(const NBoolean& node) override {
    result = builder.create<ConstantBoolOp>(loc(node.loc), node.value);
    resultType = makeTypeSpec(TypeNames::BOOL);
  }

  void visit(const NIdentifier& node) override {
    auto value = lookupVariable(node.name);
    if (value) {
      result = *value;
      const auto* type = lookupType(node.name);
      resultType =
          (type != nullptr) ? type->clone() : makeTypeSpec(TypeNames::I64);
      return;
    }

    emitError(loc(node.loc)) << "Unknown variable: " << node.name;
    result = nullptr;
  }

  void visit(const NQualifiedName& node) override {
    // Qualified name access (e.g., Math.PI)
    // Look up using mangled name
    const std::string mangled = node.getMangledName();
    auto value = lookupVariable(mangled);
    if (value) {
      result = *value;
      const auto* type = lookupType(mangled);
      resultType =
          (type != nullptr) ? type->clone() : makeTypeSpec(TypeNames::I64);
      return;
    }

    emitError(loc(node.loc))
        << "Unknown qualified name: " << node.getFullName();
    result = nullptr;
  }

  void visit(const NMethodCall& node) override {
    // Get the effective function name (mangled for qualified calls)
    std::string funcName = node.getEffectiveName();

    // Resolve imported symbol to mangled name if applicable
    auto importIt = importedSymbols.find(funcName);
    if (importIt != importedSymbols.end()) {
      funcName = importIt->second;
    }

    // Look up parameter types for literal type inference
    const std::vector<std::unique_ptr<const NTypeSpec>>* paramTypes = nullptr;
    auto paramTypesIt = functionParamTypes.find(funcName);
    if (paramTypesIt != functionParamTypes.end()) {
      paramTypes = &paramTypesIt->second;
    }

    // Collect arguments, propagating parameter types to literals
    SmallVector<Value> args;
    for (size_t i = 0; i < node.arguments.size(); ++i) {
      std::unique_ptr<const NTypeSpec> effectiveParamType;
      if (paramTypes != nullptr && i < paramTypes->size() &&
          (*paramTypes)[i] != nullptr) {
        effectiveParamType = (*paramTypes)[i]->clone();
        // For polymorphic calls, resolve type parameters to concrete bindings
        if (!node.typeBindings.empty()) {
          const std::string paramTypeName = effectiveParamType->getTypeName();
          if (polang::isTypeParameter(paramTypeName)) {
            auto bindingIt = node.typeBindings.find(paramTypeName);
            if (bindingIt != node.typeBindings.end()) {
              effectiveParamType = bindingIt->second->clone();
            }
          }
        }
      }
      if (effectiveParamType != nullptr) {
        ExpectedTypeScope scope(*this, std::move(effectiveParamType));
        node.arguments[i]->accept(*this);
      } else {
        node.arguments[i]->accept(*this);
      }
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
        resultType = funcRetTypeIt->second->clone();
      } else {
        resultType = makeTypeSpec(TypeNames::UNKNOWN);
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
        resultType = funcRetTypeIt->second->clone();
      } else {
        // Function not found - use default type for unknown function
        resultTy = getDefaultType();
        resultType = makeTypeSpec(TypeNames::UNKNOWN);
      }
    }

    // In incremental mode, emit extern FuncOp declaration if needed
    emitExternFuncDecl(funcName);

    // Check if this is a polymorphic instantiation
    if (!node.typeBindings.empty()) {
      // Build type param names and type bindings for InstantiateOp
      // Use the function's type param order from genericFunctionTypeParams
      SmallVector<StringRef> typeParamNames;
      SmallVector<Type> typeBindingTypes;

      auto paramIt = genericFunctionTypeParams.find(funcName);
      if (paramIt != genericFunctionTypeParams.end()) {
        // Use the function's declared type param order
        for (const auto& tp : paramIt->second) {
          auto bindingIt = node.typeBindings.find(tp);
          if (bindingIt != node.typeBindings.end()) {
            // Strip leading quote for MLIR storage
            StringRef paramName(tp);
            if (paramName.starts_with("'")) {
              paramName = paramName.drop_front(1);
            }
            typeParamNames.push_back(paramName);
            Type bindingType = getPolangType(*bindingIt->second);
            if (!bindingType) {
              result = nullptr;
              return;
            }
            typeBindingTypes.push_back(bindingType);
          }
        }
      } else {
        // Fallback: iterate typeBindings map directly
        for (const auto& [paramName, typeSpec] : node.typeBindings) {
          StringRef name(paramName);
          if (name.starts_with("'")) {
            name = name.drop_front(1);
          }
          typeParamNames.push_back(name);
          Type bindingType = getPolangType(*typeSpec);
          if (!bindingType) {
            result = nullptr;
            return;
          }
          typeBindingTypes.push_back(bindingType);
        }
      }

      // Resolve result type: if it's a TypeParamType, substitute with the
      // concrete binding so the InstantiateOp produces a concrete type
      if (auto paramType = dyn_cast<TypeParamType>(resultTy)) {
        StringRef paramName = paramType.getName();
        for (size_t i = 0; i < typeParamNames.size(); ++i) {
          if (typeParamNames[i] == paramName) {
            resultTy = typeBindingTypes[i];
            break;
          }
        }
      }

      auto instantiateOp = builder.create<InstantiateOp>(
          loc(node.loc), funcName, TypeRange{resultTy}, args, typeParamNames,
          typeBindingTypes);
      result = instantiateOp.getResult();
    } else {
      auto callOp = builder.create<CallOp>(loc(node.loc), funcName,
                                           TypeRange{resultTy}, args);
      result = callOp.getResult();
    }
  }

  void visit(const NUnaryOperator& node) override {
    // For TMINUS wrapping an integer literal, create the negated constant
    // directly to avoid overflow when the positive value doesn't fit in the
    // target type (e.g., -128 as i8: 128 overflows i8, but -128 fits).
    if (node.op == yy::parser::token::TMINUS) {
      if (const auto* intLit =
              dynamic_cast<const NInteger*>(node.operand.get())) {
        auto negatedValue = -static_cast<int64_t>(intLit->value);
        if (expectedLiteralType != nullptr) {
          const auto meta = getTypeMetadata(expectedLiteralType->getTypeName());
          if (meta.isInteger() || meta.isIndex()) {
            auto type = getPolangType(*expectedLiteralType);
            result = builder.create<ConstantIntegerOp>(loc(node.loc), type,
                                                       negatedValue);
            resultType = expectedLiteralType->clone();
            return;
          }
        }
        auto type = getDefaultType();
        result = builder.create<ConstantIntegerOp>(loc(node.loc), type,
                                                   negatedValue);
        resultType = makeTypeSpec(TypeNames::GENERIC_INT);
        return;
      }
    }

    node.operand->accept(*this);
    if (!result) {
      return;
    }
    Value operand = result;
    Type operandType = operand.getType();

    switch (node.op) {
    case yy::parser::token::TMINUS:
      // Unary negation (non-literal operand)
      result = builder.create<NegOp>(loc(node.loc), operandType, operand);
      // resultType remains the same as operand type
      break;
    case yy::parser::token::TNOT:
      // Logical not
      result = builder.create<NotOp>(loc(node.loc), operand);
      resultType = makeTypeSpec(TypeNames::BOOL);
      break;
    default:
      emitError(loc(node.loc)) << "Unknown unary operator: " << node.op;
      result = nullptr;
      break;
    }
  }

  void visit(const NBinaryOperator& node) override {
    node.lhs->accept(*this);
    if (!result) {
      return;
    }
    Value lhs = result;
    std::unique_ptr<const NTypeSpec> lhsType = std::move(resultType);

    // Handle short-circuit operators BEFORE evaluating RHS
    if (node.op == yy::parser::token::TLAND) {
      // a && b: if a then b else false
      Type boolType = builder.getType<BoolType>();
      auto ifOp = builder.create<IfOp>(loc(node.loc), boolType, lhs);

      // Then region: evaluate rhs and yield it
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        node.rhs->accept(*this);
        builder.create<YieldOp>(loc(node.loc), result);
      }

      // Else region: yield false
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        Value falseVal = builder.create<ConstantBoolOp>(loc(node.loc), false);
        builder.create<YieldOp>(loc(node.loc), falseVal);
      }

      result = ifOp.getResult();
      resultType = makeTypeSpec(TypeNames::BOOL);
      return;
    }
    if (node.op == yy::parser::token::TLOR) {
      // a || b: if a then true else b
      Type boolType = builder.getType<BoolType>();
      auto ifOp = builder.create<IfOp>(loc(node.loc), boolType, lhs);

      // Then region: yield true
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value trueVal = builder.create<ConstantBoolOp>(loc(node.loc), true);
        builder.create<YieldOp>(loc(node.loc), trueVal);
      }

      // Else region: evaluate rhs and yield it
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        node.rhs->accept(*this);
        builder.create<YieldOp>(loc(node.loc), result);
      }

      result = ifOp.getResult();
      resultType = makeTypeSpec(TypeNames::BOOL);
      return;
    }

    // For all other operators, evaluate RHS eagerly
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
    case yy::parser::token::TMOD:
      result = builder.create<RemOp>(loc(node.loc), arithResultTy, lhs, rhs);
      resultType = std::move(lhsType);
      break;
    // Comparison operations - result is always bool
    case yy::parser::token::TCEQ:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::eq, lhs, rhs);
      resultType = makeTypeSpec(TypeNames::BOOL);
      break;
    case yy::parser::token::TCNE:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::ne, lhs, rhs);
      resultType = makeTypeSpec(TypeNames::BOOL);
      break;
    case yy::parser::token::TCLT:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::lt, lhs, rhs);
      resultType = makeTypeSpec(TypeNames::BOOL);
      break;
    case yy::parser::token::TCLE:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::le, lhs, rhs);
      resultType = makeTypeSpec(TypeNames::BOOL);
      break;
    case yy::parser::token::TCGT:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::gt, lhs, rhs);
      resultType = makeTypeSpec(TypeNames::BOOL);
      break;
    case yy::parser::token::TCGE:
      result = builder.create<CmpOp>(loc(node.loc), CmpPredicate::ge, lhs, rhs);
      resultType = makeTypeSpec(TypeNames::BOOL);
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
    resultType = node.targetType->clone();
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
    std::unique_ptr<const NTypeSpec> ifResultType = std::move(resultType);

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

    // Track nesting so that let-bindings use local SSA values,
    // not GlobalOps, even in incremental mode.
    ++nestedScopeDepth;

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

    --nestedScopeDepth;
  }

  void visit(const NUnitLiteral& /*node*/) override {
    // Unit literal has no MLIR representation — clear result to avoid stale
    // values
    result = nullptr;
  }

  void visit(const NExpressionStatement& node) override {
    node.expression->accept(*this);
  }

  void visit(const NVariableDeclaration& node) override {
    // Get mangled variable name (includes module path)
    const std::string varName = mangledName(node.id->name);

    // In incremental mode, top-level variables become globals.
    // Nested scopes (let-expression bindings, closures) use normal SSA values.
    if (isIncremental() && isInsideEntryFunction && nestedScopeDepth == 0) {
      // Determine the type - prefer type annotation from type checker
      auto typeSpec = node.type != nullptr ? node.type->clone()
                                           : makeTypeSpec(TypeNames::I64);

      Type mlirType = getPolangType(*typeSpec);
      if (!mlirType) {
        return;
      }

      // Emit GlobalOp at module level, before the entry function.
      // Using setInsertionPoint (not setInsertionPointToStart) preserves
      // declaration order so that initializers can reference earlier globals.
      GlobalOp globalOp;
      {
        OpBuilder::InsertionGuard guard(builder);
        if (entryFuncOp) {
          builder.setInsertionPoint(entryFuncOp);
        } else {
          builder.setInsertionPointToStart(module.getBody());
        }
        globalOp = builder.create<GlobalOp>(loc(node.loc),
                                            llvm::StringRef(varName), mlirType,
                                            /*is_external=*/false);
      }

      // If there's an init expression, generate it inside the GlobalOp's
      // initializer region
      if (node.assignmentExpr != nullptr) {
        auto& initRegion = globalOp.getInitializer();
        auto* initBlock = builder.createBlock(&initRegion);

        // Save and clear immutableValues entries for same-eval globals
        // so that references inside the init region emit fresh
        // GlobalLoadOp ops instead of using SSA values from the entry
        // function (which aren't available in the init region).
        std::map<std::string, Value> savedImmutableForInit;
        for (const auto& name : currentEvalGlobals) {
          auto it = immutableValues.find(name);
          if (it != immutableValues.end()) {
            savedImmutableForInit[name] = it->second;
            immutableValues.erase(it);
          }
        }

        builder.setInsertionPointToStart(initBlock);
        {
          ExpectedTypeScope scope(*this,
                                  node.type ? node.type->clone() : nullptr);
          node.assignmentExpr->accept(*this);
        }

        if (result) {
          builder.create<YieldGlobalOp>(loc(node.loc), result);
        }

        // Restore saved immutableValues entries
        for (const auto& [name, val] : savedImmutableForInit) {
          immutableValues[name] = val;
        }
      }

      // Restore builder to end of entry function body
      builder.setInsertionPointToEnd(
          &module.getBody()->back().getRegion(0).front());

      // Track this global for SSA dominance handling in subsequent
      // init regions
      currentEvalGlobals.insert(varName);
      currentEvalGlobalTypes[varName] = mlirType;

      // Emit GlobalLoadOp and cache for use later in this eval
      auto loadOp = builder.create<GlobalLoadOp>(builder.getUnknownLoc(),
                                                 mlirType, varName);
      immutableValues[varName] = loadOp.getResult();
      typeTable[varName] = typeSpec->clone();

      result = loadOp.getResult();
      resultType = std::move(typeSpec);
      return;
    }

    // Non-incremental path: evaluate expression and use SSA value directly
    auto typeSpec = node.type != nullptr ? node.type->clone()
                                         : makeTypeSpec(TypeNames::I64);
    Value initValue = nullptr;
    if (node.assignmentExpr != nullptr) {
      {
        ExpectedTypeScope scope(*this,
                                node.type ? node.type->clone() : nullptr);
        node.assignmentExpr->accept(*this);
      }
      initValue = result;

      if (node.type == nullptr && resultType) {
        typeSpec = std::move(resultType);
      }
    }

    // All variables are immutable - store SSA value directly
    immutableValues[varName] = initValue;
    typeTable[varName] = typeSpec->clone();

    result = initValue;
    resultType = std::move(typeSpec);
  }

  void visit(const NFunctionDeclaration& node) override {
    // Save the current insertion point
    OpBuilder::InsertionGuard guard(builder);

    // Get mangled function name (includes module path)
    const std::string funcName = mangledName(node.id->name);

    // Build function type
    SmallVector<Type> argTypes;
    std::vector<std::string> argNames;
    std::vector<Type> argMLIRTypes;

    for (const auto& arg : node.arguments) {
      Type argType = arg->type ? getType(*arg->type) : getTypeFresh();
      argTypes.push_back(argType);
      argMLIRTypes.push_back(argType);
      argNames.push_back(arg->id->name);
    }

    // Add captured variables as extra parameters
    std::vector<std::string> captureNames;
    std::vector<Type> captureMLIRTypes;
    for (const auto& capture : node.captures) {
      Type captureType = capture.type ? getType(*capture.type) : getTypeFresh();
      argTypes.push_back(captureType);
      captureMLIRTypes.push_back(captureType);
      captureNames.push_back(capture.id->name);
    }

    // Store captures for call site (using mangled name)
    functionCaptures[funcName] = captureNames;

    // Store parameter types for literal type inference at call sites
    std::vector<std::unique_ptr<const NTypeSpec>> paramTypeSpecs;
    for (const auto& arg : node.arguments) {
      paramTypeSpecs.push_back(arg->type ? arg->type->clone() : nullptr);
    }
    functionParamTypes[funcName] = std::move(paramTypeSpecs);

    // Return type
    Type returnType = node.type ? getType(*node.type) : getTypeFresh();
    if (node.type != nullptr) {
      functionReturnTypes[funcName] = node.type->clone();
    }
    functionReturnMLIRTypes[funcName] = returnType;

    auto funcType = builder.getFunctionType(argTypes, {returnType});

    // Determine if the function is truly generic: typeParams is non-empty AND
    // at least one argument or the return type uses TypeParamType.
    // The AST TypeChecker may resolve all type params to concrete types
    // (e.g., in let-in expressions), leaving typeParams set but no actual
    // type parameters in the signature.
    bool isGeneric = !node.typeParams.empty();
    if (isGeneric) {
      bool hasTypeParam = isa<TypeParamType>(returnType);
      if (!hasTypeParam) {
        for (const auto& ty : argTypes) {
          if (isa<TypeParamType>(ty)) {
            hasTypeParam = true;
            break;
          }
        }
      }
      isGeneric = hasTypeParam;
    }

    // Create function at module level
    builder.setInsertionPointToEnd(module.getBody());

    SmallVector<StringRef> captureRefs;
    for (const auto& name : captureNames) {
      captureRefs.push_back(name);
    }

    Block* entryBlock = nullptr;

    if (isGeneric) {
      // Build type param names (strip leading quote for MLIR storage)
      SmallVector<StringRef> typeParamRefs;
      for (const auto& tp : node.typeParams) {
        // Type params are stored as "'a" in AST, strip to "a" for MLIR
        if (tp.size() >= 2 && tp[0] == '\'') {
          typeParamRefs.push_back(StringRef(tp).drop_front(1));
        } else {
          typeParamRefs.push_back(tp);
        }
      }

      // Build trait bounds strings (one per type param, empty if no bound)
      SmallVector<std::string> boundStrings;
      SmallVector<StringRef> boundRefs;
      for (const auto& tp : node.typeParams) {
        auto boundIt = node.typeParamBounds.find(tp);
        if (boundIt != node.typeParamBounds.end() && !boundIt->second.empty()) {
          // Use the strongest bound (take first for now)
          boundStrings.push_back(traitBoundToString(*boundIt->second.begin()));
        } else {
          boundStrings.emplace_back("");
        }
      }
      for (const auto& bs : boundStrings) {
        boundRefs.push_back(bs);
      }

      auto genericFuncOp = builder.create<GenericFuncOp>(
          loc(node.loc), funcName, funcType, typeParamRefs, boundRefs,
          ArrayRef<StringRef>(captureRefs));
      entryBlock = genericFuncOp.addEntryBlock();

      // Store type params info for call-site lookup
      genericFunctionTypeParams[funcName] = node.typeParams;
    } else {
      auto funcOp = builder.create<FuncOp>(loc(node.loc), funcName, funcType,
                                           ArrayRef<StringRef>(captureRefs));
      entryBlock = funcOp.addEntryBlock();
    }

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
        typeTable[arg->id->name] = arg->type->clone();
      }
      ++argIdx;
    }

    // Register captured variables as arguments
    for (size_t i = 0; i < node.captures.size(); ++i) {
      const auto& capture = node.captures[i];
      argValues[capture.id->name] = entryBlock->getArgument(argIdx);
      typeVarTable[capture.id->name] = captureMLIRTypes[i];
      if (capture.type != nullptr) {
        typeTable[capture.id->name] = capture.type->clone();
      }
      ++argIdx;
    }

    // Generate function body.
    // Set expectedLiteralType to the declared return type so that integer/float
    // literals in the body adopt the correct type (e.g., 3 becomes i32 when
    // the function declares () -> i32).
    {
      ExpectedTypeScope typeScope(*this,
                                  node.type ? node.type->clone() : nullptr);
      node.block->accept(*this);
    }

    // Add return
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
    const std::string moduleName = node.modulePath->getMangledName();

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
        const std::string localName = item.getEffectiveName();

        // Track the import mapping for name resolution
        importedSymbols[localName] = mangled;

        // Copy variable from mangled name to local name
        auto value = lookupVariable(mangled);
        if (value) {
          immutableValues[localName] = *value;
          const auto* type = lookupType(mangled);
          if (type != nullptr) {
            typeTable[localName] = type->clone();
          }
        }

        // Copy function info from mangled name to local name
        auto retIt = functionReturnTypes.find(mangled);
        if (retIt != functionReturnTypes.end()) {
          functionReturnTypes[localName] = retIt->second->clone();
        }
        auto retMLIRIt = functionReturnMLIRTypes.find(mangled);
        if (retMLIRIt != functionReturnMLIRTypes.end()) {
          functionReturnMLIRTypes[localName] = retMLIRIt->second;
        }
        auto paramIt = functionParamTypes.find(mangled);
        if (paramIt != functionParamTypes.end()) {
          std::vector<std::unique_ptr<const NTypeSpec>> cloned;
          cloned.reserve(paramIt->second.size());
          for (const auto& t : paramIt->second) {
            cloned.push_back(t ? t->clone() : nullptr);
          }
          functionParamTypes[localName] = std::move(cloned);
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

  void visit(const NTypeSignature& node) override {
    // Pre-register the function return type so forward-referenced calls
    // produce the correct MLIR result type.
    const NTypeSpec* innerType = node.typeExpr.get();
    if (const auto* forall = dynamic_cast<const NForallType*>(innerType)) {
      innerType = forall->innerType.get();
    }
    if (const auto* arrow = dynamic_cast<const NArrowType*>(innerType)) {
      const std::string funcName = mangledName(node.id->name);
      functionReturnTypes[funcName] = arrow->returnType->clone();
      Type retTy = getType(*arrow->returnType);
      functionReturnMLIRTypes[funcName] = retTy;
    }
    result = nullptr;
  }

private:
  OpBuilder builder;
  PolangTypeConverter typeConverter;
  ModuleOp module;
  std::string inferredType;
  std::string entryFuncName;
  std::string sourceFilename;

  // Incremental compilation support
  OptCompiledSymbols compiledSymbols;
  bool isInsideEntryFunction = false;
  int nestedScopeDepth = 0; // Tracks let-expression/function nesting
  std::set<std::string> emittedExternalGlobals; // Track emitted global decls
  std::set<std::string> emittedExternFuncs;     // Track emitted func decls
  std::set<std::string> currentEvalGlobals; // Globals defined in current eval
  std::map<std::string, Type>
      currentEvalGlobalTypes;   // Types of current-eval globals (for init
                                // regions)
  FuncOp entryFuncOp = nullptr; // Entry function for insertion ordering

  /// Check if we are in incremental mode
  [[nodiscard]] bool isIncremental() const {
    return compiledSymbols.has_value();
  }

  /// Access compiled symbols (only valid when isIncremental())
  [[nodiscard]] const CompiledSymbols& symbols() const {
    assert(compiledSymbols.has_value() &&
           "symbols() requires incremental mode");
    return compiledSymbols->get();
  }

  // Track whether any errors occurred during MLIR generation
  bool hasMLIRGenErrors = false;

  // Current result value and type
  Value result;
  std::unique_ptr<const NTypeSpec> resultType;

  // Expected type for literal expressions (set by variable declarations with
  // type annotations to propagate the annotation type to literal visitors)
  std::unique_ptr<const NTypeSpec> expectedLiteralType;

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
  std::map<std::string, Value> argValues;       // Function arguments
  std::map<std::string, Value> immutableValues; // Immutable variable SSA values
  std::map<std::string, std::unique_ptr<const NTypeSpec>>
      typeTable; // Variable types
  std::map<std::string, Type>
      typeVarTable; // Variable types as MLIR types (for type vars)
  std::map<std::string, std::vector<std::string>> functionCaptures;
  std::map<std::string, std::unique_ptr<const NTypeSpec>> functionReturnTypes;
  std::map<std::string, std::vector<std::unique_ptr<const NTypeSpec>>>
      functionParamTypes; // Function parameter types for literal inference
  std::map<std::string, Type>
      functionReturnMLIRTypes; // Function return types as MLIR types
  std::map<std::string, std::string>
      importedSymbols; // local name -> mangled name
  std::map<std::string, std::vector<std::string>>
      genericFunctionTypeParams; // func name -> type param names (e.g. "'a")

  /// RAII helper class for scoped symbol table management.
  /// Automatically saves and restores symbol tables when entering/exiting
  /// scopes.
  class SymbolTableScope {
  public:
    SymbolTableScope(MLIRGenVisitor& visitor, bool clearAllTables = false)
        : visitor(visitor), savedTypeVarTable(visitor.typeVarTable),
          savedArgValues(visitor.argValues),
          savedImmutableValues(visitor.immutableValues) {
      for (const auto& [key, val] : visitor.typeTable) {
        if (val) {
          savedTypeTable[key] = val->clone();
        }
      }
      if (clearAllTables) {
        visitor.typeTable.clear();
        visitor.typeVarTable.clear();
        visitor.argValues.clear();
        visitor.immutableValues.clear();
      }
    }

    // NOLINTNEXTLINE(modernize-use-equals-default) - destructor restores saved
    // state
    ~SymbolTableScope() {
      visitor.typeTable.clear();
      for (auto& [key, val] : savedTypeTable) {
        visitor.typeTable[key] = std::move(val);
      }
      visitor.typeVarTable = savedTypeVarTable;
      visitor.argValues = savedArgValues;
      visitor.immutableValues = savedImmutableValues;
    }

  private:
    MLIRGenVisitor& visitor;
    std::map<std::string, std::unique_ptr<const NTypeSpec>> savedTypeTable;
    std::map<std::string, Type> savedTypeVarTable;
    std::map<std::string, Value> savedArgValues;
    std::map<std::string, Value> savedImmutableValues;
  };

  /// RAII helper to set expectedLiteralType for the duration of a scope.
  /// Saves the current value and restores it on destruction.
  class ExpectedTypeScope {
  public:
    ExpectedTypeScope(MLIRGenVisitor& v, std::unique_ptr<const NTypeSpec> type)
        : visitor(v), saved(std::move(v.expectedLiteralType)) {
      visitor.expectedLiteralType = std::move(type);
    }
    ~ExpectedTypeScope() { visitor.expectedLiteralType = std::move(saved); }
    ExpectedTypeScope(const ExpectedTypeScope&) = delete;
    ExpectedTypeScope& operator=(const ExpectedTypeScope&) = delete;

  private:
    MLIRGenVisitor& visitor;
    std::unique_ptr<const NTypeSpec> saved;
  };

  /// Create MLIR location from source location
  Location loc(const SourceLocation& srcLoc) {
    if (srcLoc.isValid()) {
      return FileLineColLoc::get(builder.getContext(), sourceFilename,
                                 srcLoc.line, srcLoc.column);
    }
    return builder.getUnknownLoc();
  }

  /// Look up a variable by name in the symbol tables.
  /// Checks immutable values and function arguments.
  /// Returns nullopt if the variable is not found.
  std::optional<Value> lookupVariable(const std::string& name) {
    // Check immutable values first (SSA values, no load needed)
    auto immIt = immutableValues.find(name);
    if (immIt != immutableValues.end()) {
      return immIt->second;
    }

    // Check function arguments
    auto argIt = argValues.find(name);
    if (argIt != argValues.end()) {
      return argIt->second;
    }

    // Check current-eval globals (for use inside init regions where
    // the entry function's SSA values are not available)
    auto evalGlobalIt = currentEvalGlobalTypes.find(name);
    if (evalGlobalIt != currentEvalGlobalTypes.end()) {
      return builder
          .create<GlobalLoadOp>(builder.getUnknownLoc(), evalGlobalIt->second,
                                name)
          .getResult();
    }

    // In incremental mode, check previously compiled globals
    if (isIncremental()) {
      // Check direct global name first
      auto globalIt = symbols().globals.find(name);

      // If not found, check import mappings (e.g., "PI" → "Math$$PI")
      std::string resolvedName = name;
      if (globalIt == symbols().globals.end()) {
        auto importIt = importedSymbols.find(name);
        if (importIt != importedSymbols.end()) {
          resolvedName = importIt->second;
          globalIt = symbols().globals.find(resolvedName);
        }
      }

      if (globalIt != symbols().globals.end()) {
        // Emit external GlobalOp declaration on-demand
        emitExternalGlobalDecl(resolvedName, globalIt->second.type);

        auto globalTypeSpec = makeTypeSpec(globalIt->second.type);
        Type mlirType = getPolangType(*globalTypeSpec);
        auto loadOp = builder.create<GlobalLoadOp>(builder.getUnknownLoc(),
                                                   mlirType, resolvedName);
        immutableValues[name] = loadOp.getResult();
        typeTable[name] = std::move(globalTypeSpec);
        return loadOp.getResult();
      }
    }

    return std::nullopt;
  }

  /// Look up a variable's type by name.
  /// Returns nullptr if not found.
  [[nodiscard]] const NTypeSpec* lookupType(const std::string& name) {
    auto it = typeTable.find(name);
    if (it != typeTable.end() && it->second) {
      return it->second.get();
    }
    return nullptr;
  }

  /// Get the default type (i64) for cases where no type is specified.
  Type getDefaultType() { return typeConverter.getDefaultType(); }

  /// Get a Polang MLIR type from an NTypeSpec.
  /// Handles NNamedType only (no reference types in Polang).
  Type getPolangType(const NTypeSpec& typeSpec) {
    return typeConverter.getPolangType(typeSpec);
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
    return typeConverter.convertPolangType(polangType);
  }

  /// Emit an extern FuncOp declaration for a previously compiled function.
  /// Called on-demand when the function is called in visit(NMethodCall).
  void emitExternFuncDecl(const std::string& name) {
    // Check if we already emitted this declaration (or the function is
    // defined in this module)
    if (emittedExternFuncs.count(name) != 0U) {
      return;
    }
    if (!isIncremental()) {
      return;
    }
    auto funcIt = symbols().functions.find(name);
    if (funcIt == symbols().functions.end() || funcIt->second.isGeneric) {
      return;
    }
    emittedExternFuncs.insert(name);

    const auto& func = funcIt->second;

    // Build function type from param types + capture types
    SmallVector<Type> argTypes;
    for (const auto& paramType : func.paramTypes) {
      auto typeSpec = makeTypeSpec(paramType);
      Type mlirType = getPolangType(*typeSpec);
      if (!mlirType) {
        return;
      }
      argTypes.push_back(mlirType);
    }
    for (const auto& captureType : func.captureTypes) {
      auto typeSpec = makeTypeSpec(captureType);
      Type mlirType = getPolangType(*typeSpec);
      if (!mlirType) {
        return;
      }
      argTypes.push_back(mlirType);
    }

    auto retTypeSpec = makeTypeSpec(func.returnType);
    Type returnType = getPolangType(*retTypeSpec);
    if (!returnType) {
      return;
    }

    auto funcType = builder.getFunctionType(argTypes, {returnType});

    // Emit extern FuncOp declaration at module level
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    builder.create<FuncOp>(builder.getUnknownLoc(), name, funcType);
  }

  /// Emit an external GlobalOp declaration for a previously compiled global.
  /// Called on-demand when a global is referenced in lookupVariable().
  void emitExternalGlobalDecl(const std::string& name,
                              const std::string& typeName) {
    // Check if we already emitted this declaration
    if (emittedExternalGlobals.count(name) != 0U) {
      return;
    }
    emittedExternalGlobals.insert(name);

    auto typeSpec = makeTypeSpec(typeName);
    Type mlirType = getPolangType(*typeSpec);
    if (!mlirType) {
      return;
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    builder.create<GlobalOp>(builder.getUnknownLoc(), llvm::StringRef(name),
                             mlirType,
                             /*is_external=*/true);
  }

  /// Populate module with extern declarations from previously compiled symbols.
  /// Called in incremental mode before generating the main function.
  void populateFromCompiledSymbols() {
    builder.setInsertionPointToStart(module.getBody());

    // Note: External GlobalOp and FuncOp declarations are emitted on-demand
    // (in lookupVariable and visit(NMethodCall)) to avoid creating unused
    // declarations that cause unresolved symbols in JIT dylibs.

    // 2. Pre-populate function metadata (return types, captures, type params)
    //    for each previously compiled function. FuncOp declarations are
    //    emitted lazily when the function is actually called.
    for (const auto& [name, func] : symbols().functions) {
      if (func.isGeneric) {
        // Pre-populate generic function metadata
        genericFunctionTypeParams[name] = func.typeParams;
        functionCaptures[name] = func.captureNames;
        continue;
      }

      auto retTypeSpec = makeTypeSpec(func.returnType);
      Type returnType = getPolangType(*retTypeSpec);
      if (!returnType) {
        continue;
      }

      // Pre-populate return type, param types, and captures
      functionReturnTypes[name] = std::move(retTypeSpec);
      functionReturnMLIRTypes[name] = returnType;
      functionCaptures[name] = func.captureNames;

      // Pre-populate parameter types for literal type inference
      std::vector<std::unique_ptr<const NTypeSpec>> paramTypeSpecs;
      paramTypeSpecs.reserve(func.paramTypes.size());
      for (const auto& paramType : func.paramTypes) {
        paramTypeSpecs.push_back(makeTypeSpec(paramType));
      }
      functionParamTypes[name] = std::move(paramTypeSpecs);
    }

    // 3. Restore import mappings (local name → mangled name)
    for (const auto& [localName, mangledName] : symbols().importMappings) {
      importedSymbols[localName] = mangledName;

      // Copy function metadata from mangled name to local name
      auto retIt = functionReturnTypes.find(mangledName);
      if (retIt != functionReturnTypes.end()) {
        functionReturnTypes[localName] = retIt->second->clone();
      }
      auto retMLIRIt = functionReturnMLIRTypes.find(mangledName);
      if (retMLIRIt != functionReturnMLIRTypes.end()) {
        functionReturnMLIRTypes[localName] = retMLIRIt->second;
      }
      auto paramIt = functionParamTypes.find(mangledName);
      if (paramIt != functionParamTypes.end()) {
        std::vector<std::unique_ptr<const NTypeSpec>> cloned;
        cloned.reserve(paramIt->second.size());
        for (const auto& t : paramIt->second) {
          cloned.push_back(t ? t->clone() : nullptr);
        }
        functionParamTypes[localName] = std::move(cloned);
      }
      auto captIt = functionCaptures.find(mangledName);
      if (captIt != functionCaptures.end()) {
        functionCaptures[localName] = captIt->second;
      }

      // For imported variables, set up alias so lookupVariable resolves them
      auto globalIt = symbols().globals.find(mangledName);
      if (globalIt != symbols().globals.end()) {
        // Register the mangled global as the canonical name
        // and alias the local name to it
        typeTable[localName] = makeTypeSpec(globalIt->second.type);
      }
    }

    // 4. Re-emit GenericFuncOps by visiting stored AST nodes
    for (const auto& stmt : symbols().genericFuncAstNodes) {
      stmt->accept(*this);
    }
  }

  void generateMainFunction(const NBlock& block) {
    // Use type checker's inferred type if it's concrete, otherwise use default
    // i64
    Type returnType;
    if (inferredType == TypeNames::UNKNOWN ||
        inferredType == TypeNames::TYPEVAR || isGenericType(inferredType) ||
        polang::isTypeParameter(inferredType)) {
      // Type is unknown or generic - use default i64
      returnType = getDefaultType();
    } else {
      // Type checker found a concrete type - use it
      auto typeSpec = makeTypeSpec(inferredType);
      returnType = getPolangType(*typeSpec);
      if (!returnType) {
        return; // Stop generating if type error
      }
    }

    // Create entry function with dynamic return type
    auto funcType = builder.getFunctionType({}, {returnType});

    builder.setInsertionPointToEnd(module.getBody());
    auto entryFunc =
        builder.create<FuncOp>(loc(block.loc), entryFuncName, funcType);
    entryFuncOp = entryFunc;

    // Create entry block
    Block* entryBlock = entryFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate code for the block.
    // Set flag so visit(NVariableDeclaration) emits globals for top-level vars.
    isInsideEntryFunction = true;
    block.accept(*this);
    isInsideEntryFunction = false;

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

mlir::OwningOpRef<mlir::ModuleOp>
polang::mlirGen(mlir::MLIRContext& context, const NBlock& moduleAST,
                bool emitTypeVars, const std::string& inferredType,
                const std::string& entryFuncName,
                OptCompiledSymbols compiledSymbols) {
  // Register the Polang dialect
  context.getOrLoadDialect<PolangDialect>();

  MLIRGenVisitor generator(context, emitTypeVars, inferredType, entryFuncName,
                           "<source>", compiledSymbols);
  ModuleOp module = generator.generate(moduleAST);
  if (!module) {
    return nullptr;
  }

  return module;
}
