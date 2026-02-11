// clang-format off
#include <parser/type_checker.hpp>
#include <parser/error_reporter.hpp>
#include <parser/node.hpp>
#include <parser/operator_utils.hpp>
#include <parser/polang_types.hpp>
#include <parser/type_inference.hpp>
#include "parser.hpp"  // Must be after node.hpp for token constants
// clang-format on

#include <algorithm>
#include <iostream>
#include <set>

using polang::areTypesCompatible;
using polang::containsGenericType;
using polang::ErrorReporter;
using polang::ErrorSeverity;
using polang::formatArgCountError;
using polang::formatFuncReturnTypeError;
using polang::formatTypeMismatch;
using polang::formatUndeclaredVar;
using polang::formatUndefinedFunc;
using polang::formatVarDeclTypeError;
using polang::isArithmeticOperator;
using polang::isComparisonOperator;
using polang::isFloatType;
using polang::isGenericType;
using polang::isIntegerType;
using polang::operatorToString;
using polang::resolveAllGenericsToDefault;
using polang::resolveGenericToDefault;
using polang::TypeNames;

/// FreeVariableCollector - A visitor that identifies free variables in an
/// expression or block.
class FreeVariableCollector : public Visitor {
public:
  FreeVariableCollector(const std::set<std::string>& initialLocals)
      : localNames(initialLocals) {}

  [[nodiscard]] std::set<std::string> getReferencedNonLocals() const {
    return referencedNonLocals;
  }

  void visit(const NNamedType& /*node*/) override {}
  void visit(const NArrowType& /*node*/) override {}
  void visit(const NProductType& /*node*/) override {}
  void visit(const NTypeVar& /*node*/) override {}
  void visit(const NForallType& /*node*/) override {}
  void visit(const NInteger& /*node*/) override {}
  void visit(const NDouble& /*node*/) override {}
  void visit(const NBoolean& /*node*/) override {}

  void visit(const NIdentifier& node) override {
    if (localNames.find(node.name) == localNames.end()) {
      referencedNonLocals.insert(node.name);
    }
  }

  void visit(const NQualifiedName& /*node*/) override {}

  void visit(const NMethodCall& node) override {
    for (const auto& arg : node.arguments) {
      arg->accept(*this);
    }
  }

  void visit(const NBinaryOperator& node) override {
    node.lhs->accept(*this);
    node.rhs->accept(*this);
  }

  void visit(const NCastExpression& node) override {
    node.expression->accept(*this);
  }

  void visit(const NBlock& node) override {
    for (const auto& stmt : node.statements) {
      stmt->accept(*this);
    }
  }

  void visit(const NIfExpression& node) override {
    node.condition->accept(*this);
    node.thenExpr->accept(*this);
    node.elseExpr->accept(*this);
  }

  void visit(const NLetExpression& node) override {
    const auto savedLocals = localNames;

    for (const auto& binding : node.bindings) {
      if (binding->isFunction) {
        // Don't recurse into nested function bodies
      } else if (binding->var->assignmentExpr != nullptr) {
        binding->var->assignmentExpr->accept(*this);
      }
    }

    for (const auto& binding : node.bindings) {
      if (binding->isFunction) {
        localNames.insert(binding->func->id->name);
      } else {
        localNames.insert(binding->var->id->name);
      }
    }

    node.body->accept(*this);
    localNames = savedLocals;
  }

  void visit(const NExpressionStatement& node) override {
    node.expression->accept(*this);
  }

  void visit(const NVariableDeclaration& node) override {
    if (node.assignmentExpr != nullptr) {
      node.assignmentExpr->accept(*this);
    }
    localNames.insert(node.id->name);
  }

  void visit(const NFunctionDeclaration& node) override {
    localNames.insert(node.id->name);
  }

  void visit(const NModuleDeclaration& /*node*/) override {}
  void visit(const NImportStatement& /*node*/) override {}
  void visit(const NTypeSignature& /*node*/) override {}

private:
  std::set<std::string> localNames;
  std::set<std::string> referencedNonLocals;
};

TypeChecker::TypeChecker() : inferredType(TypeNames::I64) {}

void TypeChecker::visit(const NNamedType& node) {
  // Named types just set inferredType to their name
  inferredType = node.name;
}

void TypeChecker::visit(const NArrowType& /*node*/) {
  // Arrow types are used in type signatures, not directly visited
}

void TypeChecker::visit(const NProductType& /*node*/) {
  // Product types are used in type signatures, not directly visited
}

void TypeChecker::visit(const NTypeVar& node) {
  // Type variables in signatures set inferredType to "typevar"
  inferredType = node.getTypeName();
}

void TypeChecker::visit(const NForallType& /*node*/) {
  // Forall types are handled in applyFunctionSignature, not directly visited
}

std::string TypeChecker::mangledName(const std::string& name) const {
  if (modulePath.empty()) {
    return name;
  }
  std::string result;
  for (const auto& part : modulePath) {
    result += part + "$$";
  }
  result += name;
  return result;
}

std::vector<TypeCheckError> TypeChecker::check(const NBlock& ast) {
  errors.clear();
  localTypes.clear();
  functionSignatures.clear();
  pendingTypeSignatures.clear();
  scopeDepth = 0;
  subst = polang::Substitution();
  traitConstraints = polang::TraitConstraints();
  polang::resetUnificationVarCounter();
  ast.accept(*this);
  warnOrphanedTypeSignatures();
  return errors;
}

std::vector<TypeCheckError>
TypeChecker::checkIncremental(const NBlock& newStatements) {
  // Clear transient state but preserve persistent environment
  // (localTypes, functionSignatures)
  errors.clear();
  subst = polang::Substitution();
  traitConstraints = polang::TraitConstraints();
  newStatements.accept(*this);
  return errors;
}

TypeCheckerSnapshot TypeChecker::saveState() const {
  TypeCheckerSnapshot snapshot;
  snapshot.localTypes = localTypes;
  snapshot.functionSignatures = functionSignatures;
  snapshot.moduleExports = moduleExports;
  snapshot.moduleAliases = moduleAliases;
  snapshot.importedSymbols = importedSymbols;
  return snapshot;
}

void TypeChecker::restoreState(const TypeCheckerSnapshot& snapshot) {
  localTypes = snapshot.localTypes;
  functionSignatures = snapshot.functionSignatures;
  moduleExports = snapshot.moduleExports;
  moduleAliases = snapshot.moduleAliases;
  importedSymbols = snapshot.importedSymbols;
}

void TypeChecker::reportError(const std::string& message) {
  errors.emplace_back(message);
  auto* reporter = ErrorReporter::current();
  if (reporter != nullptr) {
    reporter->error(message);
  } else {
    std::cerr << "Type error: " << message << '\n';
  }
}

void TypeChecker::reportError(const std::string& message,
                              const SourceLocation& loc) {
  if (loc.isValid()) {
    errors.emplace_back(message, loc.line, loc.column);
    auto* reporter = ErrorReporter::current();
    if (reporter != nullptr) {
      reporter->error(message);
    } else {
      std::cerr << "Type error: " << message << " at line " << loc.line
                << ", column " << loc.column << '\n';
    }
  } else {
    reportError(message);
  }
}

void TypeChecker::visit(const NInteger& /*node*/) {
  inferredType = TypeNames::GENERIC_INT;
}

void TypeChecker::visit(const NDouble& /*node*/) {
  inferredType = TypeNames::GENERIC_FLOAT;
}

void TypeChecker::visit(const NBoolean& /*node*/) {
  inferredType = TypeNames::BOOL;
}

void TypeChecker::visit(const NIdentifier& node) {
  if (localTypes.find(node.name) == localTypes.end()) {
    reportError(formatUndeclaredVar(node.name), node.loc);
    inferredType = TypeNames::UNKNOWN;
    return;
  }
  std::string type = localTypes[node.name];
  // Resolve through substitution if it's a unification var
  if (polang::isUnificationVar(type)) {
    type = subst.apply(type);
  }
  inferredType = type;
}

void TypeChecker::visit(const NQualifiedName& node) {
  const std::string mangled = node.getMangledName();
  auto it = localTypes.find(mangled);
  if (it != localTypes.end()) {
    inferredType = it->second;
    return;
  }
  reportError("Undefined qualified name: " + node.getFullName(), node.loc);
  inferredType = TypeNames::UNKNOWN;
}

void TypeChecker::visit(const NMethodCall& node) {
  const std::string funcName = node.getEffectiveName();

  std::vector<std::string> argTypes;
  for (const auto& arg : node.arguments) {
    arg->accept(*this);
    argTypes.push_back(inferredType);
  }

  auto sigIt = functionSignatures.find(funcName);
  if (sigIt == functionSignatures.end()) {
    reportError(formatUndefinedFunc(funcName), node.loc);
    inferredType = TypeNames::UNKNOWN;
    return;
  }

  if (std::holds_alternative<polang::PolymorphicSignature>(sigIt->second)) {
    auto& mutableNode = const_cast<NMethodCall&>(node);
    instantiateCall(mutableNode, funcName,
                    std::get<polang::PolymorphicSignature>(sigIt->second),
                    argTypes);
    return;
  }

  // Monomorphic function call
  const auto& sig = std::get<polang::MonoSignature>(sigIt->second);
  const auto& paramTypes = sig.paramTypes;

  if (argTypes.size() != paramTypes.size()) {
    reportError(
        formatArgCountError(funcName, paramTypes.size(), argTypes.size()),
        node.loc);
  } else {
    // Propagate concrete types from function parameters to arguments
    for (std::size_t i = 0; i < argTypes.size(); ++i) {
      if (!isGenericType(paramTypes[i]) &&
          paramTypes[i] != TypeNames::UNKNOWN &&
          paramTypes[i] != TypeNames::TYPEVAR &&
          !polang::isTypeParameter(paramTypes[i]) &&
          !polang::isUnificationVar(paramTypes[i])) {
        propagateTypeToSource(node.arguments[i].get(), paramTypes[i]);
        node.arguments[i]->accept(*this);
        argTypes[i] = inferredType;
      }
    }

    for (std::size_t i = 0; i < argTypes.size(); ++i) {
      if (polang::isUnificationVar(argTypes[i]) ||
          polang::isUnificationVar(paramTypes[i])) {
        unifier.unify(argTypes[i], paramTypes[i], subst);
        continue;
      }
      if (argTypes[i] != TypeNames::UNKNOWN &&
          paramTypes[i] != TypeNames::UNKNOWN &&
          argTypes[i] != TypeNames::TYPEVAR &&
          paramTypes[i] != TypeNames::TYPEVAR &&
          !polang::isTypeParameter(paramTypes[i]) &&
          !areTypesCompatible(argTypes[i], paramTypes[i])) {
        reportError("Function '" + funcName + "' argument " +
                        std::to_string(i + 1) + " expects " + paramTypes[i] +
                        ", got " + resolveGenericToDefault(argTypes[i]),
                    node.loc);
      }
    }
  }

  inferredType = sig.returnType;
}

void TypeChecker::instantiateCall(NMethodCall& node,
                                  const std::string& funcName,
                                  const polang::PolymorphicSignature& scheme,
                                  const std::vector<std::string>& argTypes) {
  if (argTypes.size() != scheme.paramTypes.size()) {
    reportError(formatArgCountError(funcName, scheme.paramTypes.size(),
                                    argTypes.size()),
                node.loc);
    inferredType = TypeNames::UNKNOWN;
    return;
  }

  // Create fresh unification vars for each type parameter
  polang::Substitution callSubst;
  polang::Unifier callUnifier;
  std::map<std::string, std::string> typeParamToUniVar;
  for (const auto& tp : scheme.typeParams) {
    std::string uv = polang::freshUnificationVar();
    typeParamToUniVar[tp] = uv;
    callSubst.bind(tp, uv);
  }

  // Unify argument types with instantiated parameter types
  for (size_t i = 0; i < argTypes.size(); ++i) {
    std::string instantiatedParam = callSubst.apply(scheme.paramTypes[i]);
    std::string resolvedArg = resolveWithDefaults(argTypes[i]);
    if (resolvedArg == TypeNames::UNKNOWN) {
      continue;
    }
    if (!callUnifier.unify(instantiatedParam, resolvedArg, callSubst)) {
      reportError("Function '" + funcName + "' argument " +
                      std::to_string(i + 1) + ": type mismatch",
                  node.loc);
      inferredType = TypeNames::UNKNOWN;
      return;
    }
  }

  // Resolve all type param bindings to concrete types
  node.typeBindings.clear();
  for (const auto& tp : scheme.typeParams) {
    std::string resolved = callSubst.apply(tp);
    resolved = resolveWithDefaults(resolved);

    // Validate trait bounds
    auto boundsIt = scheme.paramBounds.find(tp);
    if (boundsIt != scheme.paramBounds.end()) {
      if (!polang::TraitConstraints::satisfies(resolved, boundsIt->second)) {
        std::string msg;
        msg += "Function '";
        msg += funcName;
        msg += "': type ";
        msg += resolved;
        msg += " does not satisfy ";
        for (auto b : boundsIt->second) {
          msg += polang::traitBoundToString(b);
          msg += " ";
        }
        msg += "for ";
        msg += tp;
        reportError(msg, node.loc);
      }
    }

    node.typeBindings[tp] = makeTypeSpec(resolved);
  }

  // Resolve return type
  std::string resolvedReturn = callSubst.apply(scheme.returnType);
  resolvedReturn = resolveWithDefaults(resolvedReturn);
  inferredType = resolvedReturn;
}

void TypeChecker::checkArithmeticBinaryOp(const NBinaryOperator& node,
                                          const std::string& lhsType,
                                          const std::string& rhsType) {
  const bool lhsIsTypevar = lhsType == TypeNames::TYPEVAR;
  const bool rhsIsTypevar = rhsType == TypeNames::TYPEVAR;
  const bool lhsIsUniVar = polang::isUnificationVar(lhsType);
  const bool rhsIsUniVar = polang::isUnificationVar(rhsType);

  // If either is a unification variable, try to unify and add Numeric bound
  if (lhsIsUniVar || rhsIsUniVar) {
    if (lhsIsUniVar) {
      traitConstraints.addBound(lhsType, polang::TraitBound::Numeric);
    }
    if (rhsIsUniVar) {
      traitConstraints.addBound(rhsType, polang::TraitBound::Numeric);
    }
    // Try to unify lhs and rhs types
    if (!unifier.unify(lhsType, rhsType, subst)) {
      reportError("Type mismatch in '" + operatorToString(node.op) +
                      "': cannot unify operand types",
                  node.loc);
    }
    // Result type is the unified type
    inferredType = subst.apply(lhsType);
    return;
  }

  if (!lhsIsTypevar && !rhsIsTypevar && !areTypesCompatible(lhsType, rhsType)) {
    reportError("Type mismatch in '" + operatorToString(node.op) +
                    "': " + resolveGenericToDefault(lhsType) + " and " +
                    resolveGenericToDefault(rhsType),
                node.loc);
  }

  if (lhsIsTypevar && !rhsIsTypevar) {
    inferredType = rhsType;
  } else {
    // Resolve generic type using the other operand as context
    inferredType = polang::resolveGenericType(lhsType, rhsType);
  }
}

void TypeChecker::checkComparisonBinaryOp(const NBinaryOperator& node,
                                          const std::string& lhsType,
                                          const std::string& rhsType) {
  const bool lhsIsTypevar = lhsType == TypeNames::TYPEVAR;
  const bool rhsIsTypevar = rhsType == TypeNames::TYPEVAR;
  const bool lhsIsUniVar = polang::isUnificationVar(lhsType);
  const bool rhsIsUniVar = polang::isUnificationVar(rhsType);

  // If either is a unification variable, add Numeric bound and unify
  if (lhsIsUniVar || rhsIsUniVar) {
    if (lhsIsUniVar) {
      traitConstraints.addBound(lhsType, polang::TraitBound::Numeric);
    }
    if (rhsIsUniVar) {
      traitConstraints.addBound(rhsType, polang::TraitBound::Numeric);
    }
    if (!unifier.unify(lhsType, rhsType, subst)) {
      reportError("Type mismatch in comparison: cannot unify operand types",
                  node.loc);
    }
    inferredType = TypeNames::BOOL;
    return;
  }

  if (!lhsIsTypevar && !rhsIsTypevar && !areTypesCompatible(lhsType, rhsType)) {
    reportError(
        "Type mismatch in comparison: " + resolveGenericToDefault(lhsType) +
            " and " + resolveGenericToDefault(rhsType),
        node.loc);
  }

  inferredType = TypeNames::BOOL;
}

void TypeChecker::visit(const NBinaryOperator& node) {
  node.lhs->accept(*this);
  const std::string lhsType = inferredType;

  node.rhs->accept(*this);
  const std::string rhsType = inferredType;

  if (lhsType == TypeNames::UNKNOWN || rhsType == TypeNames::UNKNOWN) {
    inferredType = TypeNames::UNKNOWN;
    return;
  }

  switch (polang::getOperatorCategory(node.op)) {
  case polang::OperatorCategory::Arithmetic:
    checkArithmeticBinaryOp(node, lhsType, rhsType);
    break;
  case polang::OperatorCategory::Comparison:
    checkComparisonBinaryOp(node, lhsType, rhsType);
    break;
  case polang::OperatorCategory::Unknown:
    // Unknown operator - leave type as is
    break;
  }
}

void TypeChecker::visit(const NCastExpression& node) {
  // Visit the inner expression to collect any free variables and check types
  node.expression->accept(*this);
  const std::string sourceType = polang::resolveGenericToDefault(inferredType);
  const std::string targetType = node.targetType->getTypeName();

  // Validate that both source and target are numeric (no bool casts)
  if (sourceType != polang::TypeNames::UNKNOWN &&
      sourceType != polang::TypeNames::TYPEVAR &&
      !polang::isUnificationVar(sourceType)) {
    if (!polang::isNumericType(sourceType)) {
      reportError("cannot cast " + sourceType + " to " + targetType, node.loc);
    }
    if (!polang::isNumericType(targetType)) {
      reportError("cannot cast " + sourceType + " to " + targetType, node.loc);
    }
  }

  inferredType = targetType;
}

void TypeChecker::visit(const NBlock& node) {
  // Save tracking maps for nested scope handling
  const auto savedUnresolvedGenerics = unresolvedGenerics;
  const auto savedVarDeclNodes = varDeclNodes;

  for (const auto& stmt : node.statements) {
    stmt->accept(*this);
  }

  // Resolve any remaining generic types to defaults at end of block
  resolveRemainingGenerics();

  // Restore tracking maps for outer scope
  unresolvedGenerics = savedUnresolvedGenerics;
  varDeclNodes = savedVarDeclNodes;
}

void TypeChecker::visit(const NIfExpression& node) {
  node.condition->accept(*this);
  const std::string condType = inferredType;

  if (condType != TypeNames::UNKNOWN && condType != TypeNames::BOOL &&
      condType != TypeNames::TYPEVAR && !polang::isUnificationVar(condType)) {
    reportError("If condition must be bool, got " + condType, node.loc);
  }

  // If condition is a unification var, unify it with bool
  if (polang::isUnificationVar(condType)) {
    unifier.unify(condType, TypeNames::BOOL, subst);
  }

  node.thenExpr->accept(*this);
  const std::string thenType = inferredType;

  node.elseExpr->accept(*this);
  const std::string elseType = inferredType;

  // Handle unification vars in branch types
  const bool thenIsUniVar = polang::isUnificationVar(thenType);
  const bool elseIsUniVar = polang::isUnificationVar(elseType);

  if (thenIsUniVar || elseIsUniVar) {
    // Unify the two branch types
    unifier.unify(thenType, elseType, subst);
    inferredType = subst.apply(thenType);
    return;
  }

  if (thenType != TypeNames::UNKNOWN && elseType != TypeNames::UNKNOWN &&
      thenType != TypeNames::TYPEVAR && elseType != TypeNames::TYPEVAR &&
      !areTypesCompatible(thenType, elseType)) {
    reportError("If branches have different types: " +
                    resolveGenericToDefault(thenType) + " and " +
                    resolveGenericToDefault(elseType),
                node.loc);
  }

  if (thenType == TypeNames::TYPEVAR && elseType != TypeNames::TYPEVAR &&
      elseType != TypeNames::UNKNOWN) {
    inferredType = elseType;
  } else {
    // Resolve generic type using the other branch as context
    inferredType = polang::resolveGenericType(thenType, elseType);
  }
}

void TypeChecker::collectSiblingVarTypes(
    const std::vector<std::unique_ptr<NLetBinding>>& bindings,
    std::map<std::string, std::string>& siblingTypes) {
  for (const auto& binding : bindings) {
    if (!binding->isFunction) {
      const auto& var = binding->var;
      if (var->type != nullptr) {
        siblingTypes[var->id->name] = var->type->getTypeName();
      } else if (var->assignmentExpr != nullptr) {
        var->assignmentExpr->accept(*this);
        if (inferredType != TypeNames::UNKNOWN) {
          siblingTypes[var->id->name] = inferredType;
        }
      }
    }
  }
}

void TypeChecker::typeCheckLetBindings(
    const std::vector<std::unique_ptr<NLetBinding>>& bindings,
    const std::map<std::string, std::string>& siblingTypes,
    const std::map<std::string, std::string>& savedLocals,
    std::vector<std::string>& bindingTypes,
    std::vector<std::vector<std::string>>& funcParams) {
  // NOTE: const_cast is used throughout this function to store resolved types
  // and captures back into AST nodes. See comment in
  // visit(NVariableDeclaration&).
  for (const auto& binding : bindings) {
    if (binding->isFunction) {
      const auto& func = binding->func;
      auto& mutableFunc = const_cast<NFunctionDeclaration&>(*func);

      const auto funcSavedLocals = localTypes;

      // For let-bound functions, we need to make sibling types available
      // for capture resolution. Create a merged locals map.
      std::map<std::string, std::string> mergedLocals = savedLocals;
      for (const auto& [name, type] : siblingTypes) {
        mergedLocals[name] = type;
      }

      // Use inferFunction to handle both monomorphic and polymorphic cases
      inferFunction(mutableFunc, func->id->name, mergedLocals);

      // Collect the final param types for addLetBindingsToScope
      std::vector<std::string> paramTypes;
      for (const auto& arg : func->arguments) {
        paramTypes.emplace_back(arg->type != nullptr ? arg->type->getTypeName()
                                                     : TypeNames::TYPEVAR);
      }
      funcParams.push_back(paramTypes);

      bindingTypes.emplace_back(TypeNames::FUNCTION);

      localTypes = funcSavedLocals;
    } else {
      const auto& var = binding->var;
      auto& mutableVar = const_cast<NVariableDeclaration&>(*var);

      if (var->assignmentExpr == nullptr) {
        if (var->type == nullptr) {
          reportError("Variable '" + var->id->name +
                          "' must have type annotation or initializer",
                      var->loc);
          bindingTypes.emplace_back(TypeNames::UNKNOWN);
        } else {
          bindingTypes.emplace_back(var->type->getTypeName());
        }
        continue;
      }

      var->assignmentExpr->accept(*this);
      const std::string exprType = inferredType;

      if (exprType == TypeNames::UNKNOWN) {
        if (var->type != nullptr) {
          bindingTypes.emplace_back(var->type->getTypeName());
        } else {
          bindingTypes.emplace_back(TypeNames::UNKNOWN);
        }
        continue;
      }

      if (var->type == nullptr) {
        // Resolve generic types to defaults for inferred declarations
        std::string resolvedType = resolveGenericToDefault(exprType);
        mutableVar.type = makeTypeSpec(resolvedType);
        bindingTypes.push_back(resolvedType);
      } else {
        std::string declaredType = var->type->getTypeName();
        if (!areTypesCompatible(exprType, declaredType)) {
          reportError("Variable '" + var->id->name + "' declared as " +
                          var->type->getTypeName() + " but initialized with " +
                          resolveGenericToDefault(exprType),
                      var->loc);
        }
        bindingTypes.push_back(var->type->getTypeName());
      }
    }
  }
}

void TypeChecker::addLetBindingsToScope(
    const std::vector<std::unique_ptr<NLetBinding>>& bindings,
    const std::vector<std::string>& bindingTypes,
    const std::vector<std::vector<std::string>>& funcParams) {
  std::size_t i = 0;
  std::size_t funcIdx = 0;
  for (const auto& binding : bindings) {
    if (binding->isFunction) {
      const auto& func = binding->func;
      // inferFunction already registered the signature during
      // typeCheckLetBindings; only add a fallback if it's missing
      if (functionSignatures.find(func->id->name) == functionSignatures.end()) {
        std::string returnType = func->type != nullptr
                                     ? func->type->getTypeName()
                                     : TypeNames::TYPEVAR;
        functionSignatures[func->id->name] =
            polang::MonoSignature{funcParams[funcIdx], returnType};
      }
      ++funcIdx;
    } else {
      localTypes[binding->var->id->name] = bindingTypes[i];
    }
    ++i;
  }
}

void TypeChecker::visit(const NLetExpression& node) {
  const auto savedLocals = localTypes;
  const auto savedSignatures = functionSignatures;

  // Pass 1: Collect sibling variable binding types
  std::map<std::string, std::string> siblingVarTypes;
  collectSiblingVarTypes(node.bindings, siblingVarTypes);

  // Pass 2: Type-check all binding initializers
  std::vector<std::string> bindingTypes;
  std::vector<std::vector<std::string>> funcParamTypes;
  typeCheckLetBindings(node.bindings, siblingVarTypes, savedLocals,
                       bindingTypes, funcParamTypes);

  // Pass 3: Add all bindings to scope
  addLetBindingsToScope(node.bindings, bindingTypes, funcParamTypes);

  node.body->accept(*this);

  localTypes = savedLocals;
  functionSignatures = savedSignatures;
}

void TypeChecker::visit(const NExpressionStatement& node) {
  node.expression->accept(*this);
}

void TypeChecker::typeCheckVarDeclNoInit(NVariableDeclaration& node,
                                         const std::string& varName) {
  if (node.type == nullptr) {
    reportError("Variable '" + node.id->name +
                    "' must have type annotation or initializer",
                node.loc);
    inferredType = TypeNames::UNKNOWN;
    return;
  }
  std::string baseType = node.type->getTypeName();
  localTypes[varName] = baseType;
  inferredType = baseType;
}

void TypeChecker::typeCheckVarDeclInferType(NVariableDeclaration& node,
                                            const std::string& varName,
                                            const std::string& exprType) {
  // For deferred type inference: resolve to default but track for potential
  // re-resolution
  std::string resolvedType = resolveAllGenericsToDefault(exprType);

  // Track types containing generics for later resolution
  if (containsGenericType(exprType)) {
    unresolvedGenerics[varName] = exprType;
    varDeclNodes[varName] = &node;
  }

  // Always set node.type so MLIR has valid types
  node.type = makeTypeSpec(resolvedType);
  localTypes[varName] = resolvedType;
  inferredType = resolvedType;
}

void TypeChecker::typeCheckVarDeclWithAnnotation(NVariableDeclaration& node,
                                                 const std::string& varName,
                                                 const std::string& exprType) {
  const std::string declType = node.type->getTypeName();

  const std::string& expectedType = declType;

  // If actual type could be re-resolved (source is in unresolvedGenerics) and
  // expected is concrete, propagate back. We check by trying to propagate - if
  // the source variable is in unresolvedGenerics, it will be resolved;
  // otherwise nothing happens.
  std::string actualType = exprType;
  if (!isGenericType(expectedType) && expectedType != TypeNames::TYPEVAR) {
    propagateTypeToSource(node.assignmentExpr.get(), expectedType);
    node.assignmentExpr->accept(*this);
    actualType = inferredType;
  }

  if (!areTypesCompatible(actualType, expectedType) &&
      actualType != TypeNames::TYPEVAR && expectedType != TypeNames::TYPEVAR) {
    reportError(formatVarDeclTypeError(node.id->name, declType,
                                       resolveGenericToDefault(exprType)),
                node.loc);
  }

  localTypes[varName] = declType;
  inferredType = declType;
}

void TypeChecker::visit(const NVariableDeclaration& node) {
  // NOTE: const_cast is used because the type checker needs to store resolved
  // types back into AST nodes for the MLIR generator to access. This is an
  // architectural decision - a cleaner approach would use a separate type
  // environment map shared between TypeChecker and MLIRGen.
  auto& mutableNode = const_cast<NVariableDeclaration&>(node);
  const std::string varName = mangledName(node.id->name);

  // Apply pending type signature if available
  auto sigIt = pendingTypeSignatures.find(varName);
  if (sigIt != pendingTypeSignatures.end()) {
    mutableNode.type = sigIt->second;
    pendingTypeSignatures.erase(sigIt);
  } else if (scopeDepth == 0 && node.assignmentExpr != nullptr) {
    std::cerr << "Warning: missing type signature for '" << node.id->name
              << "'\n";
  }

  if (node.assignmentExpr == nullptr) {
    typeCheckVarDeclNoInit(mutableNode, varName);
    return;
  }

  node.assignmentExpr->accept(*this);
  std::string exprType = inferredType;

  if (exprType == TypeNames::UNKNOWN) {
    if (node.type != nullptr) {
      std::string baseType = node.type->getTypeName();
      localTypes[varName] = baseType;
      inferredType = baseType;
    }
    return;
  }

  // Check if this is an inferred type case:
  // 1. No type annotation (node.type == nullptr), OR
  // 2. Type was set by previous type checker run but expression is still
  // generic
  //    (indicates the variable can be re-resolved based on context)
  bool needsInference = node.type == nullptr;
  if (!needsInference && containsGenericType(exprType)) {
    // Check if the current node type matches what we'd get from default
    // resolution
    std::string defaultType = resolveGenericToDefault(exprType);
    std::string nodeTypeName = node.type->getTypeName();
    if (nodeTypeName == defaultType) {
      // Type was set to default by previous run, allow re-resolution
      needsInference = true;
      // Clear the type so it can be re-resolved
      mutableNode.type = nullptr;
    }
  }

  if (needsInference) {
    typeCheckVarDeclInferType(mutableNode, varName, exprType);
  } else {
    typeCheckVarDeclWithAnnotation(mutableNode, varName, exprType);
  }
}

void TypeChecker::visit(const NFunctionDeclaration& node) {
  // NOTE: const_cast is used to store resolved types and captured variables
  // back into AST nodes. See comment in visit(NVariableDeclaration&).
  auto& mutableNode = const_cast<NFunctionDeclaration&>(node);

  const std::string funcName = mangledName(node.id->name);

  // Apply pending type signature if available
  auto sigIt = pendingTypeSignatures.find(funcName);
  if (sigIt != pendingTypeSignatures.end()) {
    applyFunctionSignature(mutableNode, sigIt->second);
    pendingTypeSignatures.erase(sigIt);
  } else if (scopeDepth == 0) {
    std::cerr << "Warning: missing type signature for '" << node.id->name
              << "'\n";
  }

  const auto savedLocals = localTypes;

  inferFunction(mutableNode, funcName, savedLocals);

  localTypes = savedLocals;
  inferredType =
      node.type != nullptr ? node.type->getTypeName() : TypeNames::TYPEVAR;
}

void TypeChecker::inferFunction(
    NFunctionDeclaration& node, const std::string& funcName,
    const std::map<std::string, std::string>& savedLocals) {
  // Check if function has any untyped or type-parameter parameters.
  // Type params like 'a may be left from a previous TypeChecker run
  // (e.g. MLIRGen's internal TypeChecker re-checking the accumulated AST).
  // This workaround will be removed in Phase 2 when MLIRGen no longer
  // runs its own TypeChecker.
  bool hasUntypedParams = false;
  for (const auto& arg : node.arguments) {
    if (arg->type == nullptr ||
        polang::isTypeParameter(arg->type->getTypeName())) {
      hasUntypedParams = true;
      break;
    }
  }

  // Save HM state for this function scope
  const auto savedSubst = subst;
  const auto savedTraitConstraints = traitConstraints;

  // Map from param name -> unification var (for untyped params)
  std::map<std::string, std::string> paramUniVars;

  std::set<std::string> paramNames;
  std::vector<std::string> paramTypes;
  for (const auto& arg : node.arguments) {
    paramNames.insert(arg->id->name);
    if (arg->type == nullptr ||
        polang::isTypeParameter(arg->type->getTypeName())) {
      // Assign fresh unification variable instead of TYPEVAR
      std::string uniVar = polang::freshUnificationVar();
      paramUniVars[arg->id->name] = uniVar;
      localTypes[arg->id->name] = uniVar;
      paramTypes.emplace_back(uniVar);
    } else {
      localTypes[arg->id->name] = arg->type->getTypeName();
      paramTypes.emplace_back(arg->type->getTypeName());
    }
  }

  // Collect free variables (captures)
  const std::set<std::string> freeVars =
      collectFreeVariables(*node.block, paramNames);

  node.captures.clear();
  for (const auto& varName : freeVars) {
    const auto typeIt = savedLocals.find(varName);
    if (typeIt != savedLocals.end()) {
      node.captures.emplace_back(makeTypeSpec(typeIt->second),
                                 std::make_unique<NIdentifier>(varName));
      localTypes[varName] = typeIt->second;
    }
  }

  // Pre-register signature with TYPEVAR return for recursive call support
  functionSignatures[funcName] =
      polang::MonoSignature{paramTypes, TypeNames::TYPEVAR};

  // For explicit forall: pre-unify params that share the same type variable,
  // and build the type param -> uni var mapping
  std::map<std::string, std::string> typeParamToUniVar;
  if (node.hasExplicitForall) {
    for (const auto& arg : node.arguments) {
      if (arg->type != nullptr &&
          polang::isTypeParameter(arg->type->getTypeName())) {
        std::string tp = arg->type->getTypeName();
        auto uvIt = paramUniVars.find(arg->id->name);
        if (uvIt != paramUniVars.end()) {
          auto existingIt = typeParamToUniVar.find(tp);
          if (existingIt != typeParamToUniVar.end()) {
            // Same type param → unify the uni vars
            unifier.unify(existingIt->second, uvIt->second, subst);
          } else {
            typeParamToUniVar[tp] = uvIt->second;
          }
        }
      }
    }
  }

  // Type-check function body
  ++scopeDepth;
  node.block->accept(*this);
  --scopeDepth;
  std::string bodyType = inferredType;

  // Handle explicit forall: validate bounds and build signature from declared
  // params
  if (node.hasExplicitForall) {
    // Validate: body trait constraints ⊆ declared bounds
    for (const auto& [typeParam, uniVar] : typeParamToUniVar) {
      std::string resolved = subst.apply(uniVar);
      auto accBounds = traitConstraints.getBounds(uniVar);
      // Also check bounds on the resolved var (in case of unification)
      auto resolvedBounds = traitConstraints.getBounds(resolved);
      accBounds.insert(resolvedBounds.begin(), resolvedBounds.end());

      for (auto bound : accBounds) {
        auto declIt = node.typeParamBounds.find(typeParam);
        bool isDeclared = declIt != node.typeParamBounds.end() &&
                          declIt->second.count(bound) > 0;
        if (!isDeclared) {
          reportError("type variable " + typeParam + " requires " +
                          polang::traitBoundToString(bound) +
                          " bound for arithmetic operations",
                      node.loc);
        }
      }
    }

    // Build param types using declared type param names
    std::vector<std::string> sigParamTypes;
    for (size_t i = 0; i < paramTypes.size(); ++i) {
      std::string resolved = subst.apply(paramTypes[i]);
      // Check if this uni var maps to a declared type param
      bool mapped = false;
      for (const auto& [tp, uv] : typeParamToUniVar) {
        if (subst.apply(uv) == resolved || uv == paramTypes[i]) {
          sigParamTypes.push_back(tp);
          mapped = true;
          break;
        }
      }
      if (!mapped) {
        sigParamTypes.push_back(resolveWithDefaults(resolved));
      }
      auto& mutableArg = const_cast<NVariableDeclaration&>(*node.arguments[i]);
      mutableArg.type = makeTypeSpec(sigParamTypes.back());
    }

    // Resolve return type
    std::string resolvedReturn = subst.apply(bodyType);
    std::string returnType;
    // Check if return type maps to a declared type param
    bool retMapped = false;
    for (const auto& [tp, uv] : typeParamToUniVar) {
      if (subst.apply(uv) == resolvedReturn || uv == resolvedReturn) {
        returnType = tp;
        retMapped = true;
        break;
      }
    }
    if (!retMapped) {
      if (node.type != nullptr &&
          polang::isTypeParameter(node.type->getTypeName())) {
        returnType = node.type->getTypeName();
      } else {
        returnType = resolveWithDefaults(resolvedReturn);
      }
    }
    node.type = makeTypeSpec(returnType);

    // Store polymorphic signature
    polang::PolymorphicSignature sig;
    sig.typeParams = node.typeParams;
    for (const auto& [tp, bounds] : node.typeParamBounds) {
      sig.paramBounds[tp] = bounds;
    }
    sig.paramTypes = sigParamTypes;
    sig.returnType = returnType;
    functionSignatures[funcName] = sig;

    // Restore HM state
    subst = savedSubst;
    traitConstraints = savedTraitConstraints;
    return;
  }

  if (!hasUntypedParams) {
    // Monomorphic function
    std::string returnType;
    if (node.type == nullptr) {
      if (bodyType != TypeNames::UNKNOWN && bodyType != TypeNames::TYPEVAR) {
        std::string resolvedBodyType = resolveGenericToDefault(bodyType);
        node.type = makeTypeSpec(resolvedBodyType);
        returnType = resolvedBodyType;
      } else {
        node.type = makeTypeSpec(TypeNames::TYPEVAR);
        returnType = TypeNames::TYPEVAR;
      }
    } else {
      const std::string declReturnType = node.type->getTypeName();
      if (bodyType != TypeNames::UNKNOWN && bodyType != TypeNames::TYPEVAR &&
          !areTypesCompatible(bodyType, declReturnType)) {
        reportError(polang::formatFuncReturnTypeError(
                        node.id->name, declReturnType,
                        resolveGenericToDefault(bodyType)),
                    node.loc);
      }
      returnType = declReturnType;
    }

    functionSignatures[funcName] =
        polang::MonoSignature{paramTypes, returnType};

    // Restore HM state
    subst = savedSubst;
    traitConstraints = savedTraitConstraints;
    return;
  }

  // Polymorphic function — resolve what we can via substitution
  // Apply substitution to resolve any unification vars that were unified with
  // concrete types during body type-checking
  std::string resolvedBodyType = subst.apply(bodyType);

  // Resolve params through substitution
  std::vector<std::string> resolvedParamTypes;
  resolvedParamTypes.reserve(paramTypes.size());
  for (const auto& pt : paramTypes) {
    resolvedParamTypes.push_back(subst.apply(pt));
  }

  // Check which unification vars remain unresolved (= truly polymorphic)
  // Collect the set of unification vars that are still unresolved
  std::set<std::string> unresolvedVars;
  for (const auto& rpt : resolvedParamTypes) {
    if (polang::isUnificationVar(rpt)) {
      unresolvedVars.insert(rpt);
    }
  }
  if (polang::isUnificationVar(resolvedBodyType)) {
    unresolvedVars.insert(resolvedBodyType);
  }

  if (unresolvedVars.empty()) {
    // All vars resolved to concrete types — function is monomorphic
    for (size_t i = 0; i < resolvedParamTypes.size(); ++i) {
      resolvedParamTypes[i] = resolveWithDefaults(resolvedParamTypes[i]);
      auto& mutableArg = const_cast<NVariableDeclaration&>(*node.arguments[i]);
      mutableArg.type = makeTypeSpec(resolvedParamTypes[i]);
    }
    resolvedBodyType = resolveWithDefaults(resolvedBodyType);

    std::string returnType;
    if (node.type == nullptr) {
      node.type = makeTypeSpec(resolvedBodyType);
      returnType = resolvedBodyType;
    } else {
      const std::string declReturnType = node.type->getTypeName();
      if (!areTypesCompatible(resolvedBodyType, declReturnType)) {
        reportError(polang::formatFuncReturnTypeError(
                        node.id->name, declReturnType, resolvedBodyType),
                    node.loc);
      }
      returnType = declReturnType;
    }

    functionSignatures[funcName] =
        polang::MonoSignature{resolvedParamTypes, returnType};
  } else {
    // Some vars remain unresolved — function is polymorphic
    // Name the type parameters 'a, 'b, 'c, ...
    std::map<std::string, std::string> uniVarToTypeParam;
    char paramChar = 'a';
    for (const auto& uv : unresolvedVars) {
      std::string typeParam = std::string("'") + paramChar;
      uniVarToTypeParam[uv] = typeParam;
      ++paramChar;
    }

    // Build type params and bounds
    node.typeParams.clear();
    node.typeParamBounds.clear();
    for (const auto& [uv, tp] : uniVarToTypeParam) {
      node.typeParams.push_back(tp);
      auto bounds = traitConstraints.getBounds(uv);
      if (!bounds.empty()) {
        node.typeParamBounds[tp] = bounds;
      }
    }

    // Rewrite param types: replace unification vars with type params
    for (size_t i = 0; i < resolvedParamTypes.size(); ++i) {
      auto it = uniVarToTypeParam.find(resolvedParamTypes[i]);
      if (it != uniVarToTypeParam.end()) {
        resolvedParamTypes[i] = it->second;
      } else {
        resolvedParamTypes[i] = resolveWithDefaults(resolvedParamTypes[i]);
      }
      auto& mutableArg = const_cast<NVariableDeclaration&>(*node.arguments[i]);
      mutableArg.type = makeTypeSpec(resolvedParamTypes[i]);
    }

    // Rewrite return type
    auto retIt = uniVarToTypeParam.find(resolvedBodyType);
    if (retIt != uniVarToTypeParam.end()) {
      resolvedBodyType = retIt->second;
    } else {
      resolvedBodyType = resolveWithDefaults(resolvedBodyType);
    }

    std::string returnType;
    if (node.type == nullptr) {
      node.type = makeTypeSpec(resolvedBodyType);
      returnType = resolvedBodyType;
    } else {
      const std::string declReturnType = node.type->getTypeName();
      if (!areTypesCompatible(resolvedBodyType, declReturnType)) {
        reportError(polang::formatFuncReturnTypeError(
                        node.id->name, declReturnType, resolvedBodyType),
                    node.loc);
      }
      returnType = declReturnType;
    }

    // Store polymorphic signature for instantiation
    polang::PolymorphicSignature sig;
    sig.typeParams = node.typeParams;
    for (const auto& [tp, bounds] : node.typeParamBounds) {
      sig.paramBounds[tp] = bounds;
    }
    sig.paramTypes = resolvedParamTypes;
    sig.returnType = returnType;
    functionSignatures[funcName] = sig;
  }

  // Restore HM state
  subst = savedSubst;
  traitConstraints = savedTraitConstraints;
}

std::string TypeChecker::resolveWithDefaults(const std::string& type) const {
  if (polang::isGenericIntegerType(type)) {
    return TypeNames::I64;
  }
  if (polang::isGenericFloatType(type)) {
    return TypeNames::F64;
  }
  return resolveGenericToDefault(type);
}

void TypeChecker::visit(const NModuleDeclaration& node) {
  modulePath.push_back(node.name->name);

  std::string moduleMangled;
  for (size_t i = 0; i < modulePath.size(); ++i) {
    if (i > 0) {
      moduleMangled += "$$";
    }
    moduleMangled += modulePath[i];
  }

  moduleExports[moduleMangled] =
      std::set<std::string>(node.exports.begin(), node.exports.end());

  for (const auto& member : node.members) {
    member->accept(*this);
  }

  modulePath.pop_back();
}

void TypeChecker::handleModuleImport(const NImportStatement& node) {
  const std::string moduleName = node.modulePath->getMangledName();
  moduleAliases[node.modulePath->parts.back()] = moduleName;
}

void TypeChecker::handleModuleAliasImport(const NImportStatement& node) {
  const std::string moduleName = node.modulePath->getMangledName();
  moduleAliases[node.alias] = moduleName;
}

void TypeChecker::handleItemsImport(const NImportStatement& node) {
  const std::string moduleName = node.modulePath->getMangledName();

  for (const auto& item : node.items) {
    const std::string mangledItemName = moduleName + "$$" + item.name;
    const std::string localName = item.getEffectiveName();

    auto typeIt = localTypes.find(mangledItemName);
    if (typeIt != localTypes.end()) {
      localTypes[localName] = typeIt->second;
      importedSymbols[localName] = mangledItemName;
    }

    auto sigIt = functionSignatures.find(mangledItemName);
    if (sigIt != functionSignatures.end()) {
      functionSignatures[localName] = sigIt->second;
      importedSymbols[localName] = mangledItemName;
    }
  }
}

void TypeChecker::handleWildcardImport(const NImportStatement& node) {
  const std::string moduleName = node.modulePath->getMangledName();
  const std::string prefix = moduleName + "$$";

  auto exportsIt = moduleExports.find(moduleName);
  if (exportsIt == moduleExports.end()) {
    return;
  }

  for (const auto& exportName : exportsIt->second) {
    const std::string mangledExportName = prefix + exportName;

    auto typeIt = localTypes.find(mangledExportName);
    if (typeIt != localTypes.end()) {
      localTypes[exportName] = typeIt->second;
      importedSymbols[exportName] = mangledExportName;
    }

    auto sigIt = functionSignatures.find(mangledExportName);
    if (sigIt != functionSignatures.end()) {
      functionSignatures[exportName] = sigIt->second;
      importedSymbols[exportName] = mangledExportName;
    }
  }
}

void TypeChecker::visit(const NImportStatement& node) {
  switch (node.kind) {
  case ImportKind::Module:
    handleModuleImport(node);
    break;
  case ImportKind::ModuleAlias:
    handleModuleAliasImport(node);
    break;
  case ImportKind::Items:
    handleItemsImport(node);
    break;
  case ImportKind::All:
    handleWildcardImport(node);
    break;
  }
}

void TypeChecker::visit(const NTypeSignature& node) {
  const std::string name = mangledName(node.id->name);
  pendingTypeSignatures[name] = node.typeExpr;
}

void TypeChecker::applyFunctionSignature(
    NFunctionDeclaration& node,
    const std::shared_ptr<const NTypeSpec>& signature) {
  // Check if this is a forall type signature
  const auto* forallType = dynamic_cast<const NForallType*>(signature.get());
  std::set<std::string> declaredTypeVars;
  std::shared_ptr<const NTypeSpec> innerSig = signature;

  if (forallType != nullptr) {
    node.hasExplicitForall = true;
    node.typeParams.clear();
    node.typeParamBounds.clear();

    auto& registry = polang::getTraitRegistry();

    for (const auto& tv : forallType->typeVars) {
      declaredTypeVars.insert(tv.name);
      node.typeParams.push_back(tv.name);

      if (!tv.bound.empty()) {
        if (!registry.isKnownTrait(tv.bound)) {
          reportError("unknown type class '" + tv.bound + "'");
          return;
        }
        auto traitBound = polang::stringToTraitBound(tv.bound);
        if (traitBound) {
          node.typeParamBounds[tv.name].insert(*traitBound);
        }
      }
    }

    innerSig = forallType->innerType;
  }

  // Validate type names in the signature
  std::set<std::string> usedTypeVars;
  const size_t errorsBefore = errors.size();
  validateTypeNames(innerSig.get(), declaredTypeVars, &usedTypeVars);
  if (errors.size() > errorsBefore) {
    return;
  }

  // Warn about unused type variables (only for forall signatures)
  if (forallType != nullptr) {
    for (const auto& tv : forallType->typeVars) {
      if (usedTypeVars.find(tv.name) == usedTypeVars.end()) {
        std::cerr << "Warning: unused type variable " << tv.name << "\n";
      }
    }
  }

  // Process the (inner) type as arrow type
  const auto* arrowType = dynamic_cast<const NArrowType*>(innerSig.get());
  if (arrowType == nullptr) {
    if (node.arguments.empty()) {
      // Zero-param function: non-arrow signature is just the return type
      node.type = innerSig;
      return;
    }
    reportError("type signature for '" + node.id->name +
                "' is not a function type");
    return;
  }

  // Extract parameter types
  std::vector<std::shared_ptr<const NTypeSpec>> paramTypes;
  const auto* productType =
      dynamic_cast<const NProductType*>(arrowType->paramType.get());
  if (productType != nullptr) {
    paramTypes = productType->types;
  } else {
    paramTypes.push_back(arrowType->paramType);
  }

  // Check arity matches
  if (paramTypes.size() != node.arguments.size()) {
    reportError("type signature for '" + node.id->name + "' has " +
                std::to_string(paramTypes.size()) +
                " parameters but definition has " +
                std::to_string(node.arguments.size()));
    return;
  }

  // Apply parameter types
  for (size_t i = 0; i < paramTypes.size(); ++i) {
    node.arguments[i]->type = paramTypes[i];
  }

  // Apply return type
  node.type = arrowType->returnType;
}

bool TypeChecker::validateTypeNames(
    const NTypeSpec* typeSpec, const std::set<std::string>& declaredTypeVars,
    std::set<std::string>* usedTypeVars) {
  if (typeSpec == nullptr) {
    return true;
  }

  if (const auto* named = dynamic_cast<const NNamedType*>(typeSpec)) {
    auto kind = polang::parseTypeName(named->name);
    if (!kind.has_value()) {
      reportError("unknown type '" + named->name + "'");
      return false;
    }
    return true;
  }

  if (const auto* typeVar = dynamic_cast<const NTypeVar*>(typeSpec)) {
    if (usedTypeVars != nullptr) {
      usedTypeVars->insert(typeVar->name);
    }
    if (declaredTypeVars.find(typeVar->name) == declaredTypeVars.end()) {
      reportError("undeclared type variable " + typeVar->name);
      return false;
    }
    return true;
  }

  if (const auto* arrow = dynamic_cast<const NArrowType*>(typeSpec)) {
    bool valid = validateTypeNames(arrow->paramType.get(), declaredTypeVars,
                                   usedTypeVars);
    valid = validateTypeNames(arrow->returnType.get(), declaredTypeVars,
                              usedTypeVars) &&
            valid;
    return valid;
  }

  if (const auto* product = dynamic_cast<const NProductType*>(typeSpec)) {
    bool valid = true;
    for (const auto& t : product->types) {
      valid =
          validateTypeNames(t.get(), declaredTypeVars, usedTypeVars) && valid;
    }
    return valid;
  }

  return true;
}

void TypeChecker::warnOrphanedTypeSignatures() {
  for (const auto& [name, typeExpr] : pendingTypeSignatures) {
    // Unmangle the name for display: strip module prefix (everything up to
    // and including the last "$$")
    std::string displayName = name;
    const auto pos = name.rfind("$$");
    if (pos != std::string::npos) {
      displayName = name.substr(pos + 2);
    }
    std::cerr << "Warning: type signature for '" << displayName
              << "' has no corresponding definition\n";
  }
  pendingTypeSignatures.clear();
}

std::set<std::string> TypeChecker::collectFreeVariables(
    const NBlock& block, const std::set<std::string>& localNames) const {
  FreeVariableCollector collector(localNames);
  block.accept(collector);
  return collector.getReferencedNonLocals();
}

void TypeChecker::propagateTypeToSource(const NExpression* expr,
                                        const std::string& targetType) {
  if (expr == nullptr) {
    return;
  }

  // Handle NIdentifier: resolve the variable's generic type
  if (const auto* ident = dynamic_cast<const NIdentifier*>(expr)) {
    resolveGenericVariable(ident->name, targetType);
    return;
  }

  // For other expression types (literals, etc.), no propagation needed
}

void TypeChecker::resolveGenericVariable(const std::string& varName,
                                         const std::string& concreteType) {
  // Find variable in unresolvedGenerics
  auto it = unresolvedGenerics.find(varName);
  if (it == unresolvedGenerics.end()) {
    // Variable is not tracked as having a generic type - nothing to do
    return;
  }

  const std::string& genericType = it->second;

  // Check compatibility between generic and concrete type
  if (!areTypesCompatible(genericType, concreteType)) {
    reportError("Type conflict for variable '" + varName +
                "': cannot resolve " + genericType + " to " + concreteType);
    return;
  }

  // Update localTypes with concrete type
  localTypes[varName] = concreteType;

  // Update AST node type via varDeclNodes
  auto nodeIt = varDeclNodes.find(varName);
  if (nodeIt != varDeclNodes.end() && nodeIt->second != nullptr) {
    nodeIt->second->type = makeTypeSpec(concreteType);
  }

  // Remove from unresolvedGenerics
  unresolvedGenerics.erase(it);
  varDeclNodes.erase(varName);
}

void TypeChecker::resolveRemainingGenerics() {
  // Since we now always set node.type to defaults immediately,
  // this function mainly handles updating localTypes when variables
  // weren't resolved by context propagation
  for (auto& entry : unresolvedGenerics) {
    const std::string& varName = entry.first;

    // Update localTypes with resolved type
    auto localIt = localTypes.find(varName);
    if (localIt != localTypes.end()) {
      // Resolve all generics in the type, including those in reference types
      localTypes[varName] = resolveAllGenericsToDefault(localIt->second);
    }
  }

  // Clear tracking maps
  unresolvedGenerics.clear();
  varDeclNodes.clear();
}
