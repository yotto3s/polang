// clang-format off
#include <parser/type_checker.hpp>
#include <parser/error_reporter.hpp>
#include <parser/node.hpp>
#include <parser/operator_utils.hpp>
#include <parser/polang_types.hpp>
#include "parser.hpp"  // Must be after node.hpp for token constants
// clang-format on

#include <iostream>
#include <set>

using polang::ErrorReporter;
using polang::ErrorSeverity;
using polang::isArithmeticOperator;
using polang::isComparisonOperator;
using polang::operatorToString;
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

  void visit(const NInteger& node) override {}
  void visit(const NDouble& node) override {}
  void visit(const NBoolean& node) override {}

  void visit(const NIdentifier& node) override {
    if (localNames.find(node.name) == localNames.end()) {
      referencedNonLocals.insert(node.name);
    }
  }

  void visit(const NQualifiedName& node) override {}

  void visit(const NMethodCall& node) override {
    for (const auto& arg : node.arguments) {
      arg->accept(*this);
    }
  }

  void visit(const NBinaryOperator& node) override {
    node.lhs->accept(*this);
    node.rhs->accept(*this);
  }

  void visit(const NAssignment& node) override {
    if (localNames.find(node.lhs->name) == localNames.end()) {
      referencedNonLocals.insert(node.lhs->name);
    }
    node.rhs->accept(*this);
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

  void visit(const NModuleDeclaration& node) override {}
  void visit(const NImportStatement& node) override {}

private:
  std::set<std::string> localNames;
  std::set<std::string> referencedNonLocals;
};

TypeChecker::TypeChecker() : inferredType(TypeNames::I64) {}

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
  localMutability.clear();
  functionReturnTypes.clear();
  functionParamTypes.clear();
  ast.accept(*this);
  return errors;
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

void TypeChecker::visit(const NInteger& node) { inferredType = TypeNames::I64; }

void TypeChecker::visit(const NDouble& node) { inferredType = TypeNames::F64; }

void TypeChecker::visit(const NBoolean& node) {
  inferredType = TypeNames::BOOL;
}

void TypeChecker::visit(const NIdentifier& node) {
  if (localTypes.find(node.name) == localTypes.end()) {
    reportError("Undeclared variable: " + node.name);
    inferredType = TypeNames::UNKNOWN;
    return;
  }
  inferredType = localTypes[node.name];
}

void TypeChecker::visit(const NQualifiedName& node) {
  const std::string mangled = node.mangledName();
  auto it = localTypes.find(mangled);
  if (it != localTypes.end()) {
    inferredType = it->second;
    return;
  }
  reportError("Undefined qualified name: " + node.fullName());
  inferredType = TypeNames::UNKNOWN;
}

void TypeChecker::visit(const NMethodCall& node) {
  const std::string funcName = node.effectiveName();

  std::vector<std::string> argTypes;
  for (const auto& arg : node.arguments) {
    arg->accept(*this);
    argTypes.push_back(inferredType);
  }

  const auto paramIt = functionParamTypes.find(funcName);
  if (paramIt != functionParamTypes.end()) {
    const auto& paramTypes = paramIt->second;

    if (argTypes.size() != paramTypes.size()) {
      reportError("Function '" + funcName + "' expects " +
                  std::to_string(paramTypes.size()) + " argument(s), got " +
                  std::to_string(argTypes.size()));
    } else {
      for (std::size_t i = 0; i < argTypes.size(); ++i) {
        if (argTypes[i] != TypeNames::UNKNOWN &&
            paramTypes[i] != TypeNames::UNKNOWN &&
            argTypes[i] != TypeNames::TYPEVAR &&
            paramTypes[i] != TypeNames::TYPEVAR &&
            argTypes[i] != paramTypes[i]) {
          reportError("Function '" + funcName + "' argument " +
                      std::to_string(i + 1) + " expects " + paramTypes[i] +
                      ", got " + argTypes[i]);
        }
      }
    }
  }

  if (functionReturnTypes.find(funcName) != functionReturnTypes.end()) {
    inferredType = functionReturnTypes[funcName];
  } else {
    inferredType = TypeNames::I64;
  }
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

  const bool lhsIsTypevar = lhsType == TypeNames::TYPEVAR;
  const bool rhsIsTypevar = rhsType == TypeNames::TYPEVAR;

  if (isArithmeticOperator(node.op)) {
    if (!lhsIsTypevar && !rhsIsTypevar && lhsType != rhsType) {
      reportError("Type mismatch in '" + operatorToString(node.op) +
                  "': " + lhsType + " and " + rhsType);
    }
    if (lhsIsTypevar && !rhsIsTypevar) {
      inferredType = rhsType;
    } else {
      inferredType = lhsType;
    }
  } else if (isComparisonOperator(node.op)) {
    if (!lhsIsTypevar && !rhsIsTypevar && lhsType != rhsType) {
      reportError("Type mismatch in comparison: " + lhsType + " and " +
                  rhsType);
    }
    inferredType = TypeNames::BOOL;
  }
}

void TypeChecker::visit(const NAssignment& node) {
  if (localTypes.find(node.lhs->name) == localTypes.end()) {
    reportError("Undeclared variable: " + node.lhs->name);
    inferredType = TypeNames::UNKNOWN;
    return;
  }

  if (!localMutability[node.lhs->name]) {
    reportError("Cannot reassign immutable variable: " + node.lhs->name);
    inferredType = TypeNames::UNKNOWN;
    return;
  }

  const std::string varType = localTypes[node.lhs->name];

  node.rhs->accept(*this);
  const std::string rhsType = inferredType;

  if (rhsType != TypeNames::UNKNOWN && rhsType != TypeNames::TYPEVAR &&
      varType != TypeNames::TYPEVAR && rhsType != varType) {
    reportError("Cannot assign " + rhsType + " to variable '" + node.lhs->name +
                "' of type " + varType);
  }

  inferredType = varType;
}

void TypeChecker::visit(const NBlock& node) {
  for (const auto& stmt : node.statements) {
    stmt->accept(*this);
  }
}

void TypeChecker::visit(const NIfExpression& node) {
  node.condition->accept(*this);
  const std::string condType = inferredType;

  if (condType != TypeNames::UNKNOWN && condType != TypeNames::BOOL &&
      condType != TypeNames::TYPEVAR) {
    reportError("If condition must be bool, got " + condType);
  }

  node.thenExpr->accept(*this);
  const std::string thenType = inferredType;

  node.elseExpr->accept(*this);
  const std::string elseType = inferredType;

  if (thenType != TypeNames::UNKNOWN && elseType != TypeNames::UNKNOWN &&
      thenType != TypeNames::TYPEVAR && elseType != TypeNames::TYPEVAR &&
      thenType != elseType) {
    reportError("If branches have different types: " + thenType + " and " +
                elseType);
  }

  if (thenType == TypeNames::TYPEVAR && elseType != TypeNames::TYPEVAR &&
      elseType != TypeNames::UNKNOWN) {
    inferredType = elseType;
  } else {
    inferredType = thenType;
  }
}

void TypeChecker::visit(const NLetExpression& node) {
  const auto savedLocals = localTypes;
  const auto savedMutability = localMutability;
  const auto savedFuncReturns = functionReturnTypes;
  const auto savedFuncParams = functionParamTypes;

  // Pass 1: Collect sibling variable binding types
  std::map<std::string, std::string> siblingVarTypes;
  std::map<std::string, bool> siblingVarMutability;
  for (const auto& binding : node.bindings) {
    if (!binding->isFunction) {
      const auto& var = binding->var;
      if (var->type != nullptr) {
        siblingVarTypes[var->id->name] = var->type->name;
        siblingVarMutability[var->id->name] = var->isMutable;
      } else if (var->assignmentExpr != nullptr) {
        var->assignmentExpr->accept(*this);
        if (inferredType != TypeNames::UNKNOWN) {
          siblingVarTypes[var->id->name] = inferredType;
          siblingVarMutability[var->id->name] = var->isMutable;
        }
      }
    }
  }

  // Pass 2: Type-check all binding initializers
  std::vector<std::string> bindingTypes;
  std::vector<bool> bindingMutability;
  std::vector<std::vector<std::string>> funcParamTypes;

  for (const auto& binding : node.bindings) {
    if (binding->isFunction) {
      const auto& func = binding->func;
      auto& mutableFunc = const_cast<NFunctionDeclaration&>(*func);

      const auto funcSavedLocals = localTypes;
      const auto funcSavedMutability = localMutability;

      std::set<std::string> paramNames;
      std::vector<std::string> paramTypes;
      for (const auto& arg : func->arguments) {
        paramNames.insert(arg->id->name);
        if (arg->type == nullptr) {
          auto& mutableArg = const_cast<NVariableDeclaration&>(*arg);
          mutableArg.type = std::make_unique<NIdentifier>(TypeNames::TYPEVAR);
          localTypes[arg->id->name] = TypeNames::TYPEVAR;
          localMutability[arg->id->name] = arg->isMutable;
          paramTypes.emplace_back(TypeNames::TYPEVAR);
        } else {
          localTypes[arg->id->name] = arg->type->name;
          localMutability[arg->id->name] = arg->isMutable;
          paramTypes.emplace_back(arg->type->name);
        }
      }

      funcParamTypes.push_back(paramTypes);

      const std::set<std::string> freeVars =
          collectFreeVariables(*func->block, paramNames);

      mutableFunc.captures.clear();
      for (const auto& varName : freeVars) {
        std::string varType;
        bool isMutable = false;
        bool found = false;

        if (savedLocals.find(varName) != savedLocals.end()) {
          varType = savedLocals.at(varName);
          isMutable =
              savedMutability.count(varName) > 0 && savedMutability.at(varName);
          found = true;
        } else if (siblingVarTypes.find(varName) != siblingVarTypes.end()) {
          varType = siblingVarTypes.at(varName);
          isMutable = siblingVarMutability.count(varName) > 0 &&
                      siblingVarMutability.at(varName);
          found = true;
        }

        if (found) {
          mutableFunc.captures.emplace_back(
              std::make_unique<NIdentifier>(varType),
              std::make_unique<NIdentifier>(varName), isMutable);

          localTypes[varName] = varType;
          localMutability[varName] = isMutable;
        }
      }

      func->block->accept(*this);
      const std::string bodyType = inferredType;

      if (func->type == nullptr) {
        if (bodyType != TypeNames::UNKNOWN && bodyType != TypeNames::TYPEVAR) {
          mutableFunc.type = std::make_unique<NIdentifier>(bodyType);
        } else {
          mutableFunc.type = std::make_unique<NIdentifier>(TypeNames::TYPEVAR);
        }
      } else if (bodyType != TypeNames::UNKNOWN &&
                 bodyType != TypeNames::TYPEVAR &&
                 bodyType != func->type->name) {
        reportError("Function '" + func->id->name + "' declared to return " +
                    func->type->name + " but body has type " + bodyType);
      }

      bindingTypes.emplace_back(TypeNames::FUNCTION);
      bindingMutability.push_back(false);

      localTypes = funcSavedLocals;
      localMutability = funcSavedMutability;
    } else {
      const auto& var = binding->var;
      auto& mutableVar = const_cast<NVariableDeclaration&>(*var);

      if (var->assignmentExpr == nullptr) {
        if (var->type == nullptr) {
          reportError("Variable '" + var->id->name +
                      "' must have type annotation or initializer");
          bindingTypes.emplace_back(TypeNames::UNKNOWN);
          bindingMutability.push_back(var->isMutable);
        } else {
          bindingTypes.emplace_back(var->type->name);
          bindingMutability.push_back(var->isMutable);
        }
        continue;
      }

      var->assignmentExpr->accept(*this);
      const std::string exprType = inferredType;

      if (exprType == TypeNames::UNKNOWN) {
        if (var->type != nullptr) {
          bindingTypes.emplace_back(var->type->name);
          bindingMutability.push_back(var->isMutable);
        } else {
          bindingTypes.emplace_back(TypeNames::UNKNOWN);
          bindingMutability.push_back(var->isMutable);
        }
        continue;
      }

      if (var->type == nullptr) {
        mutableVar.type = std::make_unique<NIdentifier>(exprType);
        bindingTypes.push_back(exprType);
        bindingMutability.push_back(var->isMutable);
      } else {
        if (exprType != var->type->name) {
          reportError("Variable '" + var->id->name + "' declared as " +
                      var->type->name + " but initialized with " + exprType);
        }
        bindingTypes.push_back(var->type->name);
        bindingMutability.push_back(var->isMutable);
      }
    }
  }

  // Pass 3: Add all bindings to scope
  std::size_t i = 0;
  std::size_t funcIdx = 0;
  for (const auto& binding : node.bindings) {
    if (binding->isFunction) {
      const auto& func = binding->func;
      functionParamTypes[func->id->name] = funcParamTypes[funcIdx++];
      functionReturnTypes[func->id->name] =
          func->type != nullptr ? func->type->name : TypeNames::TYPEVAR;
    } else {
      localTypes[binding->var->id->name] = bindingTypes[i];
      localMutability[binding->var->id->name] = bindingMutability[i];
    }
    ++i;
  }

  node.body->accept(*this);

  localTypes = savedLocals;
  localMutability = savedMutability;
  functionReturnTypes = savedFuncReturns;
  functionParamTypes = savedFuncParams;
}

void TypeChecker::visit(const NExpressionStatement& node) {
  node.expression->accept(*this);
}

void TypeChecker::visit(const NVariableDeclaration& node) {
  auto& mutableNode = const_cast<NVariableDeclaration&>(node);

  const std::string varName = mangledName(node.id->name);

  if (node.assignmentExpr == nullptr) {
    if (node.type == nullptr) {
      reportError("Variable '" + node.id->name +
                  "' must have type annotation or initializer");
      inferredType = TypeNames::UNKNOWN;
      return;
    }
    localTypes[varName] = node.type->name;
    localMutability[varName] = node.isMutable;
    inferredType = node.type->name;
    return;
  }

  node.assignmentExpr->accept(*this);
  const std::string exprType = inferredType;

  if (exprType == TypeNames::UNKNOWN) {
    if (node.type != nullptr) {
      localTypes[varName] = node.type->name;
      localMutability[varName] = node.isMutable;
      inferredType = node.type->name;
    }
    return;
  }

  if (node.type == nullptr) {
    mutableNode.type = std::make_unique<NIdentifier>(exprType);
    localTypes[varName] = exprType;
    localMutability[varName] = node.isMutable;
    inferredType = exprType;
  } else {
    const std::string declType = node.type->name;

    if (exprType != declType) {
      reportError("Variable '" + node.id->name + "' declared as " + declType +
                  " but initialized with " + exprType +
                  " (no implicit conversion)");
    }

    localTypes[varName] = declType;
    localMutability[varName] = node.isMutable;
    inferredType = declType;
  }
}

void TypeChecker::visit(const NFunctionDeclaration& node) {
  auto& mutableNode = const_cast<NFunctionDeclaration&>(node);

  const std::string funcName = mangledName(node.id->name);

  const auto savedLocals = localTypes;
  const auto savedMutability = localMutability;

  std::set<std::string> paramNames;
  std::vector<std::string> paramTypes;
  for (const auto& arg : node.arguments) {
    paramNames.insert(arg->id->name);
    if (arg->type == nullptr) {
      auto& mutableArg = const_cast<NVariableDeclaration&>(*arg);
      mutableArg.type = std::make_unique<NIdentifier>(TypeNames::TYPEVAR);
      localTypes[arg->id->name] = TypeNames::TYPEVAR;
      localMutability[arg->id->name] = arg->isMutable;
      paramTypes.emplace_back(TypeNames::TYPEVAR);
    } else {
      localTypes[arg->id->name] = arg->type->name;
      localMutability[arg->id->name] = arg->isMutable;
      paramTypes.emplace_back(arg->type->name);
    }
  }

  const std::set<std::string> freeVars =
      collectFreeVariables(*node.block, paramNames);

  mutableNode.captures.clear();
  for (const auto& varName : freeVars) {
    const auto typeIt = savedLocals.find(varName);
    if (typeIt != savedLocals.end()) {
      const bool isMutable =
          savedMutability.count(varName) > 0 && savedMutability.at(varName);
      mutableNode.captures.emplace_back(
          std::make_unique<NIdentifier>(typeIt->second),
          std::make_unique<NIdentifier>(varName), isMutable);

      localTypes[varName] = typeIt->second;
      localMutability[varName] = isMutable;
    }
  }

  functionParamTypes[funcName] = paramTypes;

  node.block->accept(*this);
  const std::string bodyType = inferredType;

  if (node.type == nullptr) {
    if (bodyType != TypeNames::UNKNOWN && bodyType != TypeNames::TYPEVAR) {
      mutableNode.type = std::make_unique<NIdentifier>(bodyType);
      functionReturnTypes[funcName] = bodyType;
    } else {
      mutableNode.type = std::make_unique<NIdentifier>(TypeNames::TYPEVAR);
      functionReturnTypes[funcName] = TypeNames::TYPEVAR;
    }
  } else {
    const std::string declReturnType = node.type->name;

    if (bodyType != TypeNames::UNKNOWN && bodyType != TypeNames::TYPEVAR &&
        bodyType != declReturnType) {
      reportError("Function '" + node.id->name + "' declared to return " +
                  declReturnType + " but body has type " + bodyType +
                  " (no implicit conversion)");
    }

    functionReturnTypes[funcName] = declReturnType;
  }

  localTypes = savedLocals;
  localMutability = savedMutability;
  inferredType = node.type != nullptr ? node.type->name : bodyType;
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

void TypeChecker::visit(const NImportStatement& node) {
  const std::string moduleName = node.modulePath->mangledName();

  switch (node.kind) {
  case ImportKind::Module:
    moduleAliases[node.modulePath->parts.back()] = moduleName;
    break;

  case ImportKind::ModuleAlias:
    moduleAliases[node.alias] = moduleName;
    break;

  case ImportKind::Items: {
    for (const auto& item : node.items) {
      const std::string mangledName = moduleName + "$$" + item.name;
      const std::string localName = item.effectiveName();

      auto typeIt = localTypes.find(mangledName);
      if (typeIt != localTypes.end()) {
        localTypes[localName] = typeIt->second;
        importedSymbols[localName] = mangledName;
      }

      auto retIt = functionReturnTypes.find(mangledName);
      if (retIt != functionReturnTypes.end()) {
        functionReturnTypes[localName] = retIt->second;
        auto paramIt = functionParamTypes.find(mangledName);
        if (paramIt != functionParamTypes.end()) {
          functionParamTypes[localName] = paramIt->second;
        }
        importedSymbols[localName] = mangledName;
      }
    }
    break;
  }

  case ImportKind::All: {
    const std::string prefix = moduleName + "$$";

    auto exportsIt = moduleExports.find(moduleName);
    if (exportsIt != moduleExports.end()) {
      for (const auto& exportName : exportsIt->second) {
        const std::string mangledName = prefix + exportName;

        auto typeIt = localTypes.find(mangledName);
        if (typeIt != localTypes.end()) {
          localTypes[exportName] = typeIt->second;
          importedSymbols[exportName] = mangledName;
        }

        auto retIt = functionReturnTypes.find(mangledName);
        if (retIt != functionReturnTypes.end()) {
          functionReturnTypes[exportName] = retIt->second;
          auto paramIt = functionParamTypes.find(mangledName);
          if (paramIt != functionParamTypes.end()) {
            functionParamTypes[exportName] = paramIt->second;
          }
          importedSymbols[exportName] = mangledName;
        }
      }
    }
    break;
  }
  }
}

std::set<std::string> TypeChecker::collectFreeVariables(
    const NBlock& block, const std::set<std::string>& localNames) const {
  FreeVariableCollector collector(localNames);
  block.accept(collector);
  return collector.getReferencedNonLocals();
}
