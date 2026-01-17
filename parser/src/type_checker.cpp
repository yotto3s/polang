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
///
/// This class traverses an AST subtree (typically a function body) and collects
/// all variable names that are referenced but not locally defined. These are
/// the "free variables" that must be captured by closures.
///
/// ## Algorithm
///
/// 1. Initialize with a set of known local names (e.g., function parameters)
/// 2. Traverse the AST:
///    - When an identifier is encountered, if it's not in the local set, it's a
///      free variable (add to result)
///    - When a let binding or variable declaration is encountered, add its name
///      to the local set for subsequent traversal
///    - Assignment LHS is treated as a reference (for mutable captures)
/// 3. Return the collected set of free variable names
///
/// ## Special Cases
///
/// - **Function names are not captured**: Functions are global, so when
///   visiting a method call, only the arguments are traversed, not the function
///   name itself.
/// - **Nested functions are not traversed**: Nested function bodies are not
///   recursed into because they handle their own capture analysis
///   independently.
/// - **Let bindings use parallel semantics**: In `let x = a and y = b in body`,
///   initializers `a` and `b` are processed before `x` and `y` are added to
///   scope.
///
/// ## Example
///
/// ```
/// let outer = 10 in
///   let f = fun(x: int) -> x + outer  ; 'outer' is a free variable of f
///   in f(5)
/// ```
/// When analyzing the body of `f`, `outer` is not in the local names
/// (only `x` is), so `outer` is collected as a free variable.
class FreeVariableCollector : public Visitor {
public:
  /// Construct a collector with an initial set of locally-defined names.
  /// @param initialLocals Names that should not be considered free variables
  ///        (e.g., function parameters)
  FreeVariableCollector(const std::set<std::string>& initialLocals)
      : localNames(initialLocals) {}

  [[nodiscard]] std::set<std::string> getReferencedNonLocals() const {
    return referencedNonLocals;
  }

  void visit(const NInteger& node) override {}
  void visit(const NDouble& node) override {}
  void visit(const NBoolean& node) override {}

  void visit(const NIdentifier& node) override {
    // If not locally defined, it's a free variable
    if (localNames.find(node.name) == localNames.end()) {
      referencedNonLocals.insert(node.name);
    }
  }

  void visit(const NQualifiedName& node) override {
    // Qualified names reference module members, not local variables
    // Don't treat as free variables
  }

  void visit(const NMethodCall& node) override {
    // Function name itself is not a capture (functions are global)
    // Only process arguments
    for (const auto* arg : node.arguments) {
      arg->accept(*this);
    }
  }

  void visit(const NBinaryOperator& node) override {
    node.lhs.accept(*this);
    node.rhs.accept(*this);
  }

  void visit(const NAssignment& node) override {
    // LHS is an identifier being assigned - check if it's a free variable
    if (localNames.find(node.lhs.name) == localNames.end()) {
      referencedNonLocals.insert(node.lhs.name);
    }
    node.rhs.accept(*this);
  }

  void visit(const NBlock& node) override {
    for (const auto* stmt : node.statements) {
      stmt->accept(*this);
    }
  }

  void visit(const NIfExpression& node) override {
    node.condition.accept(*this);
    node.thenExpr.accept(*this);
    node.elseExpr.accept(*this);
  }

  void visit(const NLetExpression& node) override {
    // Save current local names
    const auto savedLocals = localNames;

    // First, process initializers in current scope
    for (const auto* binding : node.bindings) {
      if (binding->isFunction) {
        // Don't recurse into nested function bodies - they have their own
        // captures
      } else if (binding->var->assignmentExpr != nullptr) {
        binding->var->assignmentExpr->accept(*this);
      }
    }

    // Add bindings to local scope for the body
    for (const auto* binding : node.bindings) {
      if (binding->isFunction) {
        localNames.insert(binding->func->id.name);
      } else {
        localNames.insert(binding->var->id.name);
      }
    }

    // Process body with extended scope
    node.body.accept(*this);

    // Restore scope
    localNames = savedLocals;
  }

  void visit(const NExpressionStatement& node) override {
    node.expression.accept(*this);
  }

  void visit(const NVariableDeclaration& node) override {
    // Process initializer
    if (node.assignmentExpr != nullptr) {
      node.assignmentExpr->accept(*this);
    }
    // Add to local scope (for subsequent statements in a block)
    localNames.insert(node.id.name);
  }

  void visit(const NFunctionDeclaration& node) override {
    // Don't recurse into nested function bodies - they will have their own
    // capture analysis Add function name to local scope
    localNames.insert(node.id.name);
  }

  void visit(const NModuleDeclaration& node) override {
    // Don't recurse into modules for free variable collection
  }

  void visit(const NImportStatement& node) override {
    // Import statements don't introduce free variables
  }

private:
  std::set<std::string> localNames;
  std::set<std::string> referencedNonLocals;
};

TypeChecker::TypeChecker() : inferredType(TypeNames::INT) {}

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
  // Use unified error reporter if available
  auto* reporter = ErrorReporter::current();
  if (reporter != nullptr) {
    reporter->error(message);
  } else {
    // Fallback to stderr (for backwards compatibility)
    std::cerr << "Type error: " << message << '\n';
  }
}

void TypeChecker::visit(const NInteger& node) { inferredType = TypeNames::INT; }

void TypeChecker::visit(const NDouble& node) {
  inferredType = TypeNames::DOUBLE;
}

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
  // Qualified name access (e.g., Math.PI)
  // Lookup using mangled name
  const std::string mangled = node.mangledName();
  auto it = localTypes.find(mangled);
  if (it != localTypes.end()) {
    inferredType = it->second;
    return;
  }
  // Not found - report error
  reportError("Undefined qualified name: " + node.fullName());
  inferredType = TypeNames::UNKNOWN;
}

void TypeChecker::visit(const NMethodCall& node) {
  // Get effective function name (mangled for qualified calls)
  const std::string funcName = node.effectiveName();

  // Collect argument types
  std::vector<std::string> argTypes;
  for (const auto* arg : node.arguments) {
    arg->accept(*this);
    argTypes.push_back(inferredType);
  }

  // Check if function is known
  const auto paramIt = functionParamTypes.find(funcName);
  if (paramIt != functionParamTypes.end()) {
    const auto& paramTypes = paramIt->second;

    // Check argument count
    if (argTypes.size() != paramTypes.size()) {
      reportError("Function '" + funcName + "' expects " +
                  std::to_string(paramTypes.size()) + " argument(s), got " +
                  std::to_string(argTypes.size()));
    } else {
      // Check each argument type
      for (std::size_t i = 0; i < argTypes.size(); ++i) {
        // Allow type variables to match any type (polymorphic inference)
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

  // Get return type from function
  if (functionReturnTypes.find(funcName) != functionReturnTypes.end()) {
    inferredType = functionReturnTypes[funcName];
  } else {
    // Unknown function, assume int
    inferredType = TypeNames::INT;
  }
}

void TypeChecker::visit(const NBinaryOperator& node) {
  // Visit left operand
  node.lhs.accept(*this);
  const std::string lhsType = inferredType;

  // Visit right operand
  node.rhs.accept(*this);
  const std::string rhsType = inferredType;

  // Check for unknown types (from undeclared variables)
  if (lhsType == TypeNames::UNKNOWN || rhsType == TypeNames::UNKNOWN) {
    inferredType = TypeNames::UNKNOWN;
    return;
  }

  // Allow typevar to match any type (MLIR will resolve later)
  const bool lhsIsTypevar = lhsType == TypeNames::TYPEVAR;
  const bool rhsIsTypevar = rhsType == TypeNames::TYPEVAR;

  if (isArithmeticOperator(node.op)) {
    // Both operands must be the same type for arithmetic (unless typevar)
    if (!lhsIsTypevar && !rhsIsTypevar && lhsType != rhsType) {
      reportError("Type mismatch in '" + operatorToString(node.op) +
                  "': " + lhsType + " and " + rhsType);
    }
    // Result type: prefer concrete type over typevar
    if (lhsIsTypevar && !rhsIsTypevar) {
      inferredType = rhsType;
    } else {
      inferredType = lhsType;
    }
  } else if (isComparisonOperator(node.op)) {
    // Comparisons also require same types (unless typevar)
    if (!lhsIsTypevar && !rhsIsTypevar && lhsType != rhsType) {
      reportError("Type mismatch in comparison: " + lhsType + " and " +
                  rhsType);
    }
    // Comparison returns bool
    inferredType = TypeNames::BOOL;
  }
}

void TypeChecker::visit(const NAssignment& node) {
  // Check variable exists
  if (localTypes.find(node.lhs.name) == localTypes.end()) {
    reportError("Undeclared variable: " + node.lhs.name);
    inferredType = TypeNames::UNKNOWN;
    return;
  }

  // Check mutability
  if (!localMutability[node.lhs.name]) {
    reportError("Cannot reassign immutable variable: " + node.lhs.name);
    inferredType = TypeNames::UNKNOWN;
    return;
  }

  const std::string varType = localTypes[node.lhs.name];

  // Check RHS type
  node.rhs.accept(*this);
  const std::string rhsType = inferredType;

  // Allow typevar to match any type (MLIR will resolve later)
  if (rhsType != TypeNames::UNKNOWN && rhsType != TypeNames::TYPEVAR &&
      varType != TypeNames::TYPEVAR && rhsType != varType) {
    reportError("Cannot assign " + rhsType + " to variable '" + node.lhs.name +
                "' of type " + varType);
  }

  inferredType = varType;
}

void TypeChecker::visit(const NBlock& node) {
  for (const auto* stmt : node.statements) {
    stmt->accept(*this);
  }
}

void TypeChecker::visit(const NIfExpression& node) {
  // Check condition
  node.condition.accept(*this);
  const std::string condType = inferredType;

  // Condition must be bool (typevar is allowed - will be inferred as bool
  // at MLIR level when type inference runs)
  if (condType != TypeNames::UNKNOWN && condType != TypeNames::BOOL &&
      condType != TypeNames::TYPEVAR) {
    reportError("If condition must be bool, got " + condType);
  }

  // Check then branch
  node.thenExpr.accept(*this);
  const std::string thenType = inferredType;

  // Check else branch
  node.elseExpr.accept(*this);
  const std::string elseType = inferredType;

  // Both branches must have same type (typevar can match any type -
  // real type checking happens at MLIR level)
  if (thenType != TypeNames::UNKNOWN && elseType != TypeNames::UNKNOWN &&
      thenType != TypeNames::TYPEVAR && elseType != TypeNames::TYPEVAR &&
      thenType != elseType) {
    reportError("If branches have different types: " + thenType + " and " +
                elseType);
  }

  // Determine the inferred type:
  // - If then is typevar and else is concrete, use else's type
  // - Otherwise use thenType (handles else-is-typevar and both-concrete cases)
  if (thenType == TypeNames::TYPEVAR && elseType != TypeNames::TYPEVAR &&
      elseType != TypeNames::UNKNOWN) {
    inferredType = elseType;
  } else {
    inferredType = thenType;
  }
}

void TypeChecker::visit(const NLetExpression& node) {
  // Save current scopes
  const auto savedLocals = localTypes;
  const auto savedMutability = localMutability;
  const auto savedFuncReturns = functionReturnTypes;
  const auto savedFuncParams = functionParamTypes;

  // Pass 1: Collect sibling variable binding types (for closure capture)
  // This allows functions to capture sibling variables in let...and expressions
  std::map<std::string, std::string> siblingVarTypes;
  std::map<std::string, bool> siblingVarMutability;
  for (const auto* binding : node.bindings) {
    if (!binding->isFunction) {
      const auto* var = binding->var;
      if (var->type != nullptr) {
        siblingVarTypes[var->id.name] = var->type->name;
        siblingVarMutability[var->id.name] = var->isMutable;
      } else if (var->assignmentExpr != nullptr) {
        // Infer type from initializer (in original scope)
        var->assignmentExpr->accept(*this);
        if (inferredType != TypeNames::UNKNOWN) {
          siblingVarTypes[var->id.name] = inferredType;
          siblingVarMutability[var->id.name] = var->isMutable;
        }
      }
    }
  }

  // Pass 2: Type-check all binding initializers in the ORIGINAL scope
  // (no new bindings visible yet - parallel/simultaneous binding semantics)
  std::vector<std::string> bindingTypes;
  std::vector<bool> bindingMutability;
  std::vector<std::vector<std::string>> funcParamTypes;

  for (const auto* binding : node.bindings) {
    if (binding->isFunction) {
      // For functions, process parameters and body but don't add to scope yet
      const auto* func = binding->func;
      auto& mutableFunc = const_cast<NFunctionDeclaration&>(*func);

      // Save scope before processing function
      const auto funcSavedLocals = localTypes;
      const auto funcSavedMutability = localMutability;

      // Collect parameter names and set up parameter types
      std::set<std::string> paramNames;
      std::vector<std::string> paramTypes;
      for (const auto* arg : func->arguments) {
        paramNames.insert(arg->id.name);
        if (arg->type == nullptr) {
          // Untyped parameter - use TYPEVAR for MLIR inference
          auto& mutableArg = const_cast<NVariableDeclaration&>(*arg);
          mutableArg.type = new NIdentifier(TypeNames::TYPEVAR);
          localTypes[arg->id.name] = TypeNames::TYPEVAR;
          localMutability[arg->id.name] = arg->isMutable;
          paramTypes.emplace_back(TypeNames::TYPEVAR);
        } else {
          localTypes[arg->id.name] = arg->type->name;
          localMutability[arg->id.name] = arg->isMutable;
          paramTypes.emplace_back(arg->type->name);
        }
      }

      // Store paramTypes for this function
      funcParamTypes.push_back(paramTypes);

      // Capture analysis: find free variables in the function body
      const std::set<std::string> freeVars =
          collectFreeVariables(func->block, paramNames);

      // Clear any existing captures and add new ones
      mutableFunc.captures.clear();
      for (const auto& varName : freeVars) {
        // Check outer scope first, then sibling variables
        std::string varType;
        bool isMutable = false;
        bool found = false;

        if (savedLocals.find(varName) != savedLocals.end()) {
          // Found in outer scope
          varType = savedLocals.at(varName);
          isMutable =
              savedMutability.count(varName) > 0 && savedMutability.at(varName);
          found = true;
        } else if (siblingVarTypes.find(varName) != siblingVarTypes.end()) {
          // Found in sibling bindings
          varType = siblingVarTypes.at(varName);
          isMutable = siblingVarMutability.count(varName) > 0 &&
                      siblingVarMutability.at(varName);
          found = true;
        }

        if (found) {
          auto* captureId = new NIdentifier(varName);
          auto* captureType = new NIdentifier(varType);
          auto* capture = new NVariableDeclaration(captureType, *captureId,
                                                   nullptr, isMutable);
          mutableFunc.captures.push_back(capture);

          // Add captured variable to local scope for body type checking
          localTypes[varName] = varType;
          localMutability[varName] = isMutable;
        }
        // If not found, it will be caught as "undeclared variable"
      }

      // Type-check function body
      func->block.accept(*this);
      const std::string bodyType = inferredType;

      // Handle return type: use annotation, infer from body, or use TYPEVAR
      if (func->type == nullptr) {
        if (bodyType != TypeNames::UNKNOWN && bodyType != TypeNames::TYPEVAR) {
          mutableFunc.type = new NIdentifier(bodyType);
        } else {
          // Let MLIR infer the return type
          mutableFunc.type = new NIdentifier(TypeNames::TYPEVAR);
        }
      } else if (bodyType != TypeNames::UNKNOWN &&
                 bodyType != TypeNames::TYPEVAR &&
                 bodyType != func->type->name) {
        reportError("Function '" + func->id.name + "' declared to return " +
                    func->type->name + " but body has type " + bodyType);
      }

      // Placeholder for function binding type and mutability
      bindingTypes.emplace_back(TypeNames::FUNCTION);
      bindingMutability.push_back(false);

      // Restore scope (remove parameters)
      localTypes = funcSavedLocals;
      localMutability = funcSavedMutability;
    } else {
      // Variable binding: type-check initializer without adding to scope
      const auto* var = binding->var;
      auto& mutableVar = const_cast<NVariableDeclaration&>(*var);

      if (var->assignmentExpr == nullptr) {
        if (var->type == nullptr) {
          reportError("Variable '" + var->id.name +
                      "' must have type annotation or initializer");
          bindingTypes.emplace_back(TypeNames::UNKNOWN);
          bindingMutability.push_back(var->isMutable);
        } else {
          bindingTypes.emplace_back(var->type->name);
          bindingMutability.push_back(var->isMutable);
        }
        continue;
      }

      // Type-check initializer in current scope (no new bindings visible)
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
        // Infer type from expression
        mutableVar.type = new NIdentifier(exprType);
        bindingTypes.push_back(exprType);
        bindingMutability.push_back(var->isMutable);
      } else {
        // Validate type annotation
        if (exprType != var->type->name) {
          reportError("Variable '" + var->id.name + "' declared as " +
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
  for (const auto* binding : node.bindings) {
    if (binding->isFunction) {
      const auto* func = binding->func;
      functionParamTypes[func->id.name] = funcParamTypes[funcIdx++];
      functionReturnTypes[func->id.name] =
          func->type != nullptr ? func->type->name : TypeNames::TYPEVAR;
    } else {
      localTypes[binding->var->id.name] = bindingTypes[i];
      localMutability[binding->var->id.name] = bindingMutability[i];
    }
    ++i;
  }

  // Process body with all bindings in scope
  node.body.accept(*this);

  // Restore previous scopes
  localTypes = savedLocals;
  localMutability = savedMutability;
  functionReturnTypes = savedFuncReturns;
  functionParamTypes = savedFuncParams;
}

void TypeChecker::visit(const NExpressionStatement& node) {
  node.expression.accept(*this);
}

void TypeChecker::visit(const NVariableDeclaration& node) {
  // Cast away const to allow setting inferred type
  auto& mutableNode = const_cast<NVariableDeclaration&>(node);

  // Get mangled name (includes module path if inside a module)
  const std::string varName = mangledName(node.id.name);

  if (node.assignmentExpr == nullptr) {
    // No initializer - must have type annotation
    if (node.type == nullptr) {
      reportError("Variable '" + node.id.name +
                  "' must have type annotation or initializer");
      inferredType = TypeNames::UNKNOWN;
      return;
    }
    localTypes[varName] = node.type->name;
    localMutability[varName] = node.isMutable;
    inferredType = node.type->name;
    return;
  }

  // Infer expression type
  node.assignmentExpr->accept(*this);
  const std::string exprType = inferredType;

  if (exprType == TypeNames::UNKNOWN) {
    // Error already reported for expression
    if (node.type != nullptr) {
      localTypes[varName] = node.type->name;
      localMutability[varName] = node.isMutable;
      inferredType = node.type->name;
    }
    return;
  }

  if (node.type == nullptr) {
    // No type annotation - infer from expression
    mutableNode.type = new NIdentifier(exprType);
    localTypes[varName] = exprType;
    localMutability[varName] = node.isMutable;
    inferredType = exprType;
  } else {
    // Type annotation present - validate (no coercion!)
    const std::string declType = node.type->name;

    if (exprType != declType) {
      reportError("Variable '" + node.id.name + "' declared as " + declType +
                  " but initialized with " + exprType +
                  " (no implicit conversion)");
    }

    localTypes[varName] = declType;
    localMutability[varName] = node.isMutable;
    inferredType = declType;
  }
}

void TypeChecker::visit(const NFunctionDeclaration& node) {
  // Cast away const to allow setting inferred type
  auto& mutableNode = const_cast<NFunctionDeclaration&>(node);

  // Get mangled name (includes module path if inside a module)
  const std::string funcName = mangledName(node.id.name);

  // Save current scope
  const auto savedLocals = localTypes;
  const auto savedMutability = localMutability;

  // Collect parameter names and set up parameter types
  std::set<std::string> paramNames;
  std::vector<std::string> paramTypes;
  for (const auto* arg : node.arguments) {
    paramNames.insert(arg->id.name);
    if (arg->type == nullptr) {
      // Untyped parameter - use TYPEVAR for MLIR inference
      auto& mutableArg = const_cast<NVariableDeclaration&>(*arg);
      mutableArg.type = new NIdentifier(TypeNames::TYPEVAR);
      localTypes[arg->id.name] = TypeNames::TYPEVAR;
      localMutability[arg->id.name] = arg->isMutable;
      paramTypes.emplace_back(TypeNames::TYPEVAR);
    } else {
      localTypes[arg->id.name] = arg->type->name;
      localMutability[arg->id.name] = arg->isMutable;
      paramTypes.emplace_back(arg->type->name);
    }
  }

  // Capture analysis: find free variables in the function body
  const std::set<std::string> freeVars =
      collectFreeVariables(node.block, paramNames);

  // Clear any existing captures and add new ones
  mutableNode.captures.clear();
  for (const auto& varName : freeVars) {
    // Check if this free variable exists in the outer scope
    const auto typeIt = savedLocals.find(varName);
    if (typeIt != savedLocals.end()) {
      // Create a capture entry for this variable
      auto* captureId = new NIdentifier(varName);
      auto* captureType = new NIdentifier(typeIt->second);
      const bool isMutable =
          savedMutability.count(varName) > 0 && savedMutability.at(varName);
      auto* capture =
          new NVariableDeclaration(captureType, *captureId, nullptr, isMutable);
      mutableNode.captures.push_back(capture);

      // Add captured variable to local scope for body type checking
      localTypes[varName] = typeIt->second;
      localMutability[varName] = isMutable;
    }
    // If not in outer scope, it will be caught as "undeclared variable" later
  }

  // Store parameter types for this function (using mangled name)
  functionParamTypes[funcName] = paramTypes;

  // Check function body
  node.block.accept(*this);
  const std::string bodyType = inferredType;

  if (node.type == nullptr) {
    // Return type not annotated - infer from body or use TYPEVAR
    if (bodyType != TypeNames::UNKNOWN && bodyType != TypeNames::TYPEVAR) {
      mutableNode.type = new NIdentifier(bodyType);
      functionReturnTypes[funcName] = bodyType;
    } else {
      // Let MLIR infer the return type
      mutableNode.type = new NIdentifier(TypeNames::TYPEVAR);
      functionReturnTypes[funcName] = TypeNames::TYPEVAR;
    }
  } else {
    // Return type annotated - validate
    const std::string declReturnType = node.type->name;

    if (bodyType != TypeNames::UNKNOWN && bodyType != TypeNames::TYPEVAR &&
        bodyType != declReturnType) {
      reportError("Function '" + node.id.name + "' declared to return " +
                  declReturnType + " but body has type " + bodyType +
                  " (no implicit conversion)");
    }

    functionReturnTypes[funcName] = declReturnType;
  }

  // Restore previous scope
  localTypes = savedLocals;
  localMutability = savedMutability;
  inferredType = node.type != nullptr ? node.type->name : bodyType;
}

void TypeChecker::visit(const NModuleDeclaration& node) {
  // Push module name onto path for name mangling
  modulePath.push_back(node.name.name);

  // Build module mangled name
  std::string moduleMangled;
  for (size_t i = 0; i < modulePath.size(); ++i) {
    if (i > 0) {
      moduleMangled += "$$";
    }
    moduleMangled += modulePath[i];
  }

  // Register module exports
  moduleExports[moduleMangled] =
      std::set<std::string>(node.exports.begin(), node.exports.end());

  // Visit all members (they will register with mangled names)
  for (const auto* member : node.members) {
    member->accept(*this);
  }

  // Pop module name from path
  modulePath.pop_back();
}

void TypeChecker::visit(const NImportStatement& node) {
  const std::string moduleName = node.modulePath.mangledName();

  switch (node.kind) {
  case ImportKind::Module:
    // import Math - just register the module alias as itself
    // Access will be through qualified names like Math.add
    moduleAliases[node.modulePath.parts.back()] = moduleName;
    break;

  case ImportKind::ModuleAlias:
    // import Math as M - register the alias
    moduleAliases[node.alias] = moduleName;
    break;

  case ImportKind::Items: {
    // from Math import add, PI - import specific items
    for (const auto& item : node.items) {
      const std::string mangledName = moduleName + "$$" + item.name;
      const std::string localName = item.effectiveName();

      // Check if the symbol exists in localTypes or functionParamTypes
      auto typeIt = localTypes.find(mangledName);
      if (typeIt != localTypes.end()) {
        // It's a variable - create an alias
        localTypes[localName] = typeIt->second;
        importedSymbols[localName] = mangledName;
      }

      auto retIt = functionReturnTypes.find(mangledName);
      if (retIt != functionReturnTypes.end()) {
        // It's a function - create aliases for return type and params
        functionReturnTypes[localName] = retIt->second;
        auto paramIt = functionParamTypes.find(mangledName);
        if (paramIt != functionParamTypes.end()) {
          functionParamTypes[localName] = paramIt->second;
        }
        importedSymbols[localName] = mangledName;
      }

      // If neither found, the error will be caught when the symbol is used
    }
    break;
  }

  case ImportKind::All: {
    // from Math import * - import all exports
    // Find all symbols that start with moduleName$$
    const std::string prefix = moduleName + "$$";

    // Check moduleExports for what's exported
    auto exportsIt = moduleExports.find(moduleName);
    if (exportsIt != moduleExports.end()) {
      for (const auto& exportName : exportsIt->second) {
        const std::string mangledName = prefix + exportName;

        // Import variable if exists
        auto typeIt = localTypes.find(mangledName);
        if (typeIt != localTypes.end()) {
          localTypes[exportName] = typeIt->second;
          importedSymbols[exportName] = mangledName;
        }

        // Import function if exists
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
