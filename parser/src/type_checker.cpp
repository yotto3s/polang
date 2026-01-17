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
  /// @param initial_locals Names that should not be considered free variables
  ///        (e.g., function parameters)
  FreeVariableCollector(const std::set<std::string>& initial_locals)
      : local_names_(initial_locals) {}

  std::set<std::string> getReferencedNonLocals() const {
    return referenced_non_locals_;
  }

  void visit(const NInteger& node) override {}
  void visit(const NDouble& node) override {}
  void visit(const NBoolean& node) override {}

  void visit(const NIdentifier& node) override {
    // If not locally defined, it's a free variable
    if (local_names_.find(node.name) == local_names_.end()) {
      referenced_non_locals_.insert(node.name);
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
    if (local_names_.find(node.lhs.name) == local_names_.end()) {
      referenced_non_locals_.insert(node.lhs.name);
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
    const auto saved_locals = local_names_;

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
        local_names_.insert(binding->func->id.name);
      } else {
        local_names_.insert(binding->var->id.name);
      }
    }

    // Process body with extended scope
    node.body.accept(*this);

    // Restore scope
    local_names_ = saved_locals;
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
    local_names_.insert(node.id.name);
  }

  void visit(const NFunctionDeclaration& node) override {
    // Don't recurse into nested function bodies - they will have their own
    // capture analysis Add function name to local scope
    local_names_.insert(node.id.name);
  }

  void visit(const NModuleDeclaration& node) override {
    // Don't recurse into modules for free variable collection
  }

  void visit(const NImportStatement& node) override {
    // Import statements don't introduce free variables
  }

private:
  std::set<std::string> local_names_;
  std::set<std::string> referenced_non_locals_;
};

TypeChecker::TypeChecker() : inferred_type_(TypeNames::INT) {}

std::string TypeChecker::mangledName(const std::string& name) const {
  if (module_path_.empty()) {
    return name;
  }
  std::string result;
  for (const auto& part : module_path_) {
    result += part + "$$";
  }
  result += name;
  return result;
}

std::vector<TypeCheckError> TypeChecker::check(const NBlock& ast) {
  errors_.clear();
  local_types_.clear();
  local_mutability_.clear();
  function_return_types_.clear();
  function_param_types_.clear();
  ast.accept(*this);
  return errors_;
}

void TypeChecker::reportError(const std::string& message) {
  errors_.push_back(TypeCheckError(message));
  // Use unified error reporter if available
  auto* reporter = ErrorReporter::current();
  if (reporter) {
    reporter->error(message);
  } else {
    // Fallback to stderr (for backwards compatibility)
    std::cerr << "Type error: " << message << std::endl;
  }
}

void TypeChecker::visit(const NInteger& node) {
  inferred_type_ = TypeNames::INT;
}

void TypeChecker::visit(const NDouble& node) {
  inferred_type_ = TypeNames::DOUBLE;
}

void TypeChecker::visit(const NBoolean& node) {
  inferred_type_ = TypeNames::BOOL;
}

void TypeChecker::visit(const NIdentifier& node) {
  if (local_types_.find(node.name) == local_types_.end()) {
    reportError("Undeclared variable: " + node.name);
    inferred_type_ = TypeNames::UNKNOWN;
    return;
  }
  inferred_type_ = local_types_[node.name];
}

void TypeChecker::visit(const NQualifiedName& node) {
  // Qualified name access (e.g., Math.PI)
  // Lookup using mangled name
  const std::string mangled = node.mangledName();
  auto it = local_types_.find(mangled);
  if (it != local_types_.end()) {
    inferred_type_ = it->second;
    return;
  }
  // Not found - report error
  reportError("Undefined qualified name: " + node.fullName());
  inferred_type_ = TypeNames::UNKNOWN;
}

void TypeChecker::visit(const NMethodCall& node) {
  // Get effective function name (mangled for qualified calls)
  const std::string func_name = node.effectiveName();

  // Collect argument types
  std::vector<std::string> arg_types;
  for (const auto* arg : node.arguments) {
    arg->accept(*this);
    arg_types.push_back(inferred_type_);
  }

  // Check if function is known
  const auto param_it = function_param_types_.find(func_name);
  if (param_it != function_param_types_.end()) {
    const auto& param_types = param_it->second;

    // Check argument count
    if (arg_types.size() != param_types.size()) {
      reportError("Function '" + func_name + "' expects " +
                  std::to_string(param_types.size()) + " argument(s), got " +
                  std::to_string(arg_types.size()));
    } else {
      // Check each argument type
      for (std::size_t i = 0; i < arg_types.size(); ++i) {
        // Allow type variables to match any type (polymorphic inference)
        if (arg_types[i] != TypeNames::UNKNOWN &&
            param_types[i] != TypeNames::UNKNOWN &&
            arg_types[i] != TypeNames::TYPEVAR &&
            param_types[i] != TypeNames::TYPEVAR &&
            arg_types[i] != param_types[i]) {
          reportError("Function '" + func_name + "' argument " +
                      std::to_string(i + 1) + " expects " + param_types[i] +
                      ", got " + arg_types[i]);
        }
      }
    }
  }

  // Get return type from function
  if (function_return_types_.find(func_name) != function_return_types_.end()) {
    inferred_type_ = function_return_types_[func_name];
  } else {
    // Unknown function, assume int
    inferred_type_ = TypeNames::INT;
  }
}

void TypeChecker::visit(const NBinaryOperator& node) {
  // Visit left operand
  node.lhs.accept(*this);
  const std::string lhs_type = inferred_type_;

  // Visit right operand
  node.rhs.accept(*this);
  const std::string rhs_type = inferred_type_;

  // Check for unknown types (from undeclared variables)
  if (lhs_type == TypeNames::UNKNOWN || rhs_type == TypeNames::UNKNOWN) {
    inferred_type_ = TypeNames::UNKNOWN;
    return;
  }

  // Allow typevar to match any type (MLIR will resolve later)
  const bool lhs_is_typevar = lhs_type == TypeNames::TYPEVAR;
  const bool rhs_is_typevar = rhs_type == TypeNames::TYPEVAR;

  if (isArithmeticOperator(node.op)) {
    // Both operands must be the same type for arithmetic (unless typevar)
    if (!lhs_is_typevar && !rhs_is_typevar && lhs_type != rhs_type) {
      reportError("Type mismatch in '" + operatorToString(node.op) +
                  "': " + lhs_type + " and " + rhs_type);
    }
    // Result type: prefer concrete type over typevar
    if (lhs_is_typevar && !rhs_is_typevar) {
      inferred_type_ = rhs_type;
    } else {
      inferred_type_ = lhs_type;
    }
  } else if (isComparisonOperator(node.op)) {
    // Comparisons also require same types (unless typevar)
    if (!lhs_is_typevar && !rhs_is_typevar && lhs_type != rhs_type) {
      reportError("Type mismatch in comparison: " + lhs_type + " and " +
                  rhs_type);
    }
    // Comparison returns bool
    inferred_type_ = TypeNames::BOOL;
  }
}

void TypeChecker::visit(const NAssignment& node) {
  // Check variable exists
  if (local_types_.find(node.lhs.name) == local_types_.end()) {
    reportError("Undeclared variable: " + node.lhs.name);
    inferred_type_ = TypeNames::UNKNOWN;
    return;
  }

  // Check mutability
  if (!local_mutability_[node.lhs.name]) {
    reportError("Cannot reassign immutable variable: " + node.lhs.name);
    inferred_type_ = TypeNames::UNKNOWN;
    return;
  }

  const std::string var_type = local_types_[node.lhs.name];

  // Check RHS type
  node.rhs.accept(*this);
  const std::string rhs_type = inferred_type_;

  // Allow typevar to match any type (MLIR will resolve later)
  if (rhs_type != TypeNames::UNKNOWN && rhs_type != TypeNames::TYPEVAR &&
      var_type != TypeNames::TYPEVAR && rhs_type != var_type) {
    reportError("Cannot assign " + rhs_type + " to variable '" + node.lhs.name +
                "' of type " + var_type);
  }

  inferred_type_ = var_type;
}

void TypeChecker::visit(const NBlock& node) {
  for (const auto* stmt : node.statements) {
    stmt->accept(*this);
  }
}

void TypeChecker::visit(const NIfExpression& node) {
  // Check condition
  node.condition.accept(*this);
  const std::string cond_type = inferred_type_;

  // Condition must be bool (typevar is allowed - will be inferred as bool
  // at MLIR level when type inference runs)
  if (cond_type != TypeNames::UNKNOWN && cond_type != TypeNames::BOOL &&
      cond_type != TypeNames::TYPEVAR) {
    reportError("If condition must be bool, got " + cond_type);
  }

  // Check then branch
  node.thenExpr.accept(*this);
  const std::string then_type = inferred_type_;

  // Check else branch
  node.elseExpr.accept(*this);
  const std::string else_type = inferred_type_;

  // Both branches must have same type (typevar can match any type -
  // real type checking happens at MLIR level)
  if (then_type != TypeNames::UNKNOWN && else_type != TypeNames::UNKNOWN &&
      then_type != TypeNames::TYPEVAR && else_type != TypeNames::TYPEVAR &&
      then_type != else_type) {
    reportError("If branches have different types: " + then_type + " and " +
                else_type);
  }

  // Determine the inferred type:
  // - If one branch is typevar, use the other branch's concrete type
  // - Otherwise use then_type
  if (then_type == TypeNames::TYPEVAR && else_type != TypeNames::TYPEVAR &&
      else_type != TypeNames::UNKNOWN) {
    inferred_type_ = else_type;
  } else if (else_type == TypeNames::TYPEVAR &&
             then_type != TypeNames::TYPEVAR &&
             then_type != TypeNames::UNKNOWN) {
    inferred_type_ = then_type;
  } else {
    inferred_type_ = then_type;
  }
}

void TypeChecker::visit(const NLetExpression& node) {
  // Save current scopes
  const auto saved_locals = local_types_;
  const auto saved_mutability = local_mutability_;
  const auto saved_func_returns = function_return_types_;
  const auto saved_func_params = function_param_types_;

  // Pass 1: Collect sibling variable binding types (for closure capture)
  // This allows functions to capture sibling variables in let...and expressions
  std::map<std::string, std::string> sibling_var_types;
  std::map<std::string, bool> sibling_var_mutability;
  for (const auto* binding : node.bindings) {
    if (!binding->isFunction) {
      const auto* var = binding->var;
      if (var->type != nullptr) {
        sibling_var_types[var->id.name] = var->type->name;
        sibling_var_mutability[var->id.name] = var->isMutable;
      } else if (var->assignmentExpr != nullptr) {
        // Infer type from initializer (in original scope)
        var->assignmentExpr->accept(*this);
        if (inferred_type_ != TypeNames::UNKNOWN) {
          sibling_var_types[var->id.name] = inferred_type_;
          sibling_var_mutability[var->id.name] = var->isMutable;
        }
      }
    }
  }

  // Pass 2: Type-check all binding initializers in the ORIGINAL scope
  // (no new bindings visible yet - parallel/simultaneous binding semantics)
  std::vector<std::string> binding_types;
  std::vector<bool> binding_mutability;
  std::vector<std::vector<std::string>> func_param_types;

  for (const auto* binding : node.bindings) {
    if (binding->isFunction) {
      // For functions, process parameters and body but don't add to scope yet
      const auto* func = binding->func;
      NFunctionDeclaration& mutable_func =
          const_cast<NFunctionDeclaration&>(*func);

      // Save scope before processing function
      const auto func_saved_locals = local_types_;
      const auto func_saved_mutability = local_mutability_;

      // Collect parameter names and set up parameter types
      std::set<std::string> param_names;
      std::vector<std::string> param_types;
      for (const auto* arg : func->arguments) {
        param_names.insert(arg->id.name);
        if (arg->type == nullptr) {
          // Untyped parameter - use TYPEVAR for MLIR inference
          auto& mutable_arg = const_cast<NVariableDeclaration&>(*arg);
          mutable_arg.type = new NIdentifier(TypeNames::TYPEVAR);
          local_types_[arg->id.name] = TypeNames::TYPEVAR;
          local_mutability_[arg->id.name] = arg->isMutable;
          param_types.push_back(TypeNames::TYPEVAR);
        } else {
          local_types_[arg->id.name] = arg->type->name;
          local_mutability_[arg->id.name] = arg->isMutable;
          param_types.push_back(arg->type->name);
        }
      }

      // Store param_types for this function
      func_param_types.push_back(param_types);

      // Capture analysis: find free variables in the function body
      const std::set<std::string> free_vars =
          collectFreeVariables(func->block, param_names);

      // Clear any existing captures and add new ones
      mutable_func.captures.clear();
      for (const auto& var_name : free_vars) {
        // Check outer scope first, then sibling variables
        std::string var_type;
        bool is_mutable = false;
        bool found = false;

        if (saved_locals.find(var_name) != saved_locals.end()) {
          // Found in outer scope
          var_type = saved_locals.at(var_name);
          is_mutable = saved_mutability.count(var_name) > 0 &&
                       saved_mutability.at(var_name);
          found = true;
        } else if (sibling_var_types.find(var_name) !=
                   sibling_var_types.end()) {
          // Found in sibling bindings
          var_type = sibling_var_types.at(var_name);
          is_mutable = sibling_var_mutability.count(var_name) > 0 &&
                       sibling_var_mutability.at(var_name);
          found = true;
        }

        if (found) {
          auto* capture_id = new NIdentifier(var_name);
          auto* capture_type = new NIdentifier(var_type);
          auto* capture = new NVariableDeclaration(capture_type, *capture_id,
                                                   nullptr, is_mutable);
          mutable_func.captures.push_back(capture);

          // Add captured variable to local scope for body type checking
          local_types_[var_name] = var_type;
          local_mutability_[var_name] = is_mutable;
        }
        // If not found, it will be caught as "undeclared variable"
      }

      // Type-check function body
      func->block.accept(*this);
      const std::string body_type = inferred_type_;

      // Handle return type: use annotation, infer from body, or use TYPEVAR
      if (func->type == nullptr) {
        if (body_type != TypeNames::UNKNOWN &&
            body_type != TypeNames::TYPEVAR) {
          mutable_func.type = new NIdentifier(body_type);
        } else {
          // Let MLIR infer the return type
          mutable_func.type = new NIdentifier(TypeNames::TYPEVAR);
        }
      } else if (body_type != TypeNames::UNKNOWN &&
                 body_type != TypeNames::TYPEVAR &&
                 body_type != func->type->name) {
        reportError("Function '" + func->id.name + "' declared to return " +
                    func->type->name + " but body has type " + body_type);
      }

      // Placeholder for function binding type and mutability
      binding_types.push_back(TypeNames::FUNCTION);
      binding_mutability.push_back(false);

      // Restore scope (remove parameters)
      local_types_ = func_saved_locals;
      local_mutability_ = func_saved_mutability;
    } else {
      // Variable binding: type-check initializer without adding to scope
      const auto* var = binding->var;
      NVariableDeclaration& mutable_var =
          const_cast<NVariableDeclaration&>(*var);

      if (var->assignmentExpr == nullptr) {
        if (var->type == nullptr) {
          reportError("Variable '" + var->id.name +
                      "' must have type annotation or initializer");
          binding_types.push_back(TypeNames::UNKNOWN);
          binding_mutability.push_back(var->isMutable);
        } else {
          binding_types.push_back(var->type->name);
          binding_mutability.push_back(var->isMutable);
        }
        continue;
      }

      // Type-check initializer in current scope (no new bindings visible)
      var->assignmentExpr->accept(*this);
      const std::string expr_type = inferred_type_;

      if (expr_type == TypeNames::UNKNOWN) {
        if (var->type != nullptr) {
          binding_types.push_back(var->type->name);
          binding_mutability.push_back(var->isMutable);
        } else {
          binding_types.push_back(TypeNames::UNKNOWN);
          binding_mutability.push_back(var->isMutable);
        }
        continue;
      }

      if (var->type == nullptr) {
        // Infer type from expression
        mutable_var.type = new NIdentifier(expr_type);
        binding_types.push_back(expr_type);
        binding_mutability.push_back(var->isMutable);
      } else {
        // Validate type annotation
        if (expr_type != var->type->name) {
          reportError("Variable '" + var->id.name + "' declared as " +
                      var->type->name + " but initialized with " + expr_type);
        }
        binding_types.push_back(var->type->name);
        binding_mutability.push_back(var->isMutable);
      }
    }
  }

  // Pass 3: Add all bindings to scope
  std::size_t i = 0;
  std::size_t func_idx = 0;
  for (const auto* binding : node.bindings) {
    if (binding->isFunction) {
      const auto* func = binding->func;
      function_param_types_[func->id.name] = func_param_types[func_idx++];
      function_return_types_[func->id.name] =
          func->type ? func->type->name : TypeNames::TYPEVAR;
    } else {
      local_types_[binding->var->id.name] = binding_types[i];
      local_mutability_[binding->var->id.name] = binding_mutability[i];
    }
    ++i;
  }

  // Process body with all bindings in scope
  node.body.accept(*this);

  // Restore previous scopes
  local_types_ = saved_locals;
  local_mutability_ = saved_mutability;
  function_return_types_ = saved_func_returns;
  function_param_types_ = saved_func_params;
}

void TypeChecker::visit(const NExpressionStatement& node) {
  node.expression.accept(*this);
}

void TypeChecker::visit(const NVariableDeclaration& node) {
  // Cast away const to allow setting inferred type
  NVariableDeclaration& mutable_node = const_cast<NVariableDeclaration&>(node);

  // Get mangled name (includes module path if inside a module)
  const std::string var_name = mangledName(node.id.name);

  if (node.assignmentExpr == nullptr) {
    // No initializer - must have type annotation
    if (node.type == nullptr) {
      reportError("Variable '" + node.id.name +
                  "' must have type annotation or initializer");
      inferred_type_ = TypeNames::UNKNOWN;
      return;
    }
    local_types_[var_name] = node.type->name;
    local_mutability_[var_name] = node.isMutable;
    inferred_type_ = node.type->name;
    return;
  }

  // Infer expression type
  node.assignmentExpr->accept(*this);
  const std::string expr_type = inferred_type_;

  if (expr_type == TypeNames::UNKNOWN) {
    // Error already reported for expression
    if (node.type != nullptr) {
      local_types_[var_name] = node.type->name;
      local_mutability_[var_name] = node.isMutable;
      inferred_type_ = node.type->name;
    }
    return;
  }

  if (node.type == nullptr) {
    // No type annotation - infer from expression
    mutable_node.type = new NIdentifier(expr_type);
    local_types_[var_name] = expr_type;
    local_mutability_[var_name] = node.isMutable;
    inferred_type_ = expr_type;
  } else {
    // Type annotation present - validate (no coercion!)
    const std::string decl_type = node.type->name;

    if (expr_type != decl_type) {
      reportError("Variable '" + node.id.name + "' declared as " + decl_type +
                  " but initialized with " + expr_type +
                  " (no implicit conversion)");
    }

    local_types_[var_name] = decl_type;
    local_mutability_[var_name] = node.isMutable;
    inferred_type_ = decl_type;
  }
}

void TypeChecker::visit(const NFunctionDeclaration& node) {
  // Cast away const to allow setting inferred type
  NFunctionDeclaration& mutable_node = const_cast<NFunctionDeclaration&>(node);

  // Get mangled name (includes module path if inside a module)
  const std::string func_name = mangledName(node.id.name);

  // Save current scope
  const auto saved_locals = local_types_;
  const auto saved_mutability = local_mutability_;

  // Collect parameter names and set up parameter types
  std::set<std::string> param_names;
  std::vector<std::string> param_types;
  for (const auto* arg : node.arguments) {
    param_names.insert(arg->id.name);
    if (arg->type == nullptr) {
      // Untyped parameter - use TYPEVAR for MLIR inference
      auto& mutable_arg = const_cast<NVariableDeclaration&>(*arg);
      mutable_arg.type = new NIdentifier(TypeNames::TYPEVAR);
      local_types_[arg->id.name] = TypeNames::TYPEVAR;
      local_mutability_[arg->id.name] = arg->isMutable;
      param_types.push_back(TypeNames::TYPEVAR);
    } else {
      local_types_[arg->id.name] = arg->type->name;
      local_mutability_[arg->id.name] = arg->isMutable;
      param_types.push_back(arg->type->name);
    }
  }

  // Capture analysis: find free variables in the function body
  const std::set<std::string> free_vars =
      collectFreeVariables(node.block, param_names);

  // Clear any existing captures and add new ones
  mutable_node.captures.clear();
  for (const auto& var_name : free_vars) {
    // Check if this free variable exists in the outer scope
    const auto type_it = saved_locals.find(var_name);
    if (type_it != saved_locals.end()) {
      // Create a capture entry for this variable
      auto* capture_id = new NIdentifier(var_name);
      auto* capture_type = new NIdentifier(type_it->second);
      const bool is_mutable =
          saved_mutability.count(var_name) > 0 && saved_mutability.at(var_name);
      auto* capture = new NVariableDeclaration(capture_type, *capture_id,
                                               nullptr, is_mutable);
      mutable_node.captures.push_back(capture);

      // Add captured variable to local scope for body type checking
      local_types_[var_name] = type_it->second;
      local_mutability_[var_name] = is_mutable;
    }
    // If not in outer scope, it will be caught as "undeclared variable" later
  }

  // Store parameter types for this function (using mangled name)
  function_param_types_[func_name] = param_types;

  // Check function body
  node.block.accept(*this);
  const std::string body_type = inferred_type_;

  if (node.type == nullptr) {
    // Return type not annotated - infer from body or use TYPEVAR
    if (body_type != TypeNames::UNKNOWN && body_type != TypeNames::TYPEVAR) {
      mutable_node.type = new NIdentifier(body_type);
      function_return_types_[func_name] = body_type;
    } else {
      // Let MLIR infer the return type
      mutable_node.type = new NIdentifier(TypeNames::TYPEVAR);
      function_return_types_[func_name] = TypeNames::TYPEVAR;
    }
  } else {
    // Return type annotated - validate
    const std::string decl_return_type = node.type->name;

    if (body_type != TypeNames::UNKNOWN && body_type != TypeNames::TYPEVAR &&
        body_type != decl_return_type) {
      reportError("Function '" + node.id.name + "' declared to return " +
                  decl_return_type + " but body has type " + body_type +
                  " (no implicit conversion)");
    }

    function_return_types_[func_name] = decl_return_type;
  }

  // Restore previous scope
  local_types_ = saved_locals;
  local_mutability_ = saved_mutability;
  inferred_type_ = node.type ? node.type->name : body_type;
}

void TypeChecker::visit(const NModuleDeclaration& node) {
  // Push module name onto path for name mangling
  module_path_.push_back(node.name.name);

  // Build module mangled name
  std::string module_mangled;
  for (size_t i = 0; i < module_path_.size(); ++i) {
    if (i > 0)
      module_mangled += "$$";
    module_mangled += module_path_[i];
  }

  // Register module exports
  module_exports_[module_mangled] =
      std::set<std::string>(node.exports.begin(), node.exports.end());

  // Visit all members (they will register with mangled names)
  for (const auto* member : node.members) {
    member->accept(*this);
  }

  // Pop module name from path
  module_path_.pop_back();
}

void TypeChecker::visit(const NImportStatement& node) {
  const std::string module_name = node.modulePath.mangledName();

  switch (node.kind) {
  case ImportKind::Module:
    // import Math - just register the module alias as itself
    // Access will be through qualified names like Math.add
    module_aliases_[node.modulePath.parts.back()] = module_name;
    break;

  case ImportKind::ModuleAlias:
    // import Math as M - register the alias
    module_aliases_[node.alias] = module_name;
    break;

  case ImportKind::Items: {
    // from Math import add, PI - import specific items
    for (const auto& item : node.items) {
      const std::string mangled_name = module_name + "$$" + item.name;
      const std::string local_name = item.effectiveName();

      // Check if the symbol exists in local_types_ or function_param_types_
      auto type_it = local_types_.find(mangled_name);
      if (type_it != local_types_.end()) {
        // It's a variable - create an alias
        local_types_[local_name] = type_it->second;
        imported_symbols_[local_name] = mangled_name;
      }

      auto ret_it = function_return_types_.find(mangled_name);
      if (ret_it != function_return_types_.end()) {
        // It's a function - create aliases for return type and params
        function_return_types_[local_name] = ret_it->second;
        auto param_it = function_param_types_.find(mangled_name);
        if (param_it != function_param_types_.end()) {
          function_param_types_[local_name] = param_it->second;
        }
        imported_symbols_[local_name] = mangled_name;
      }

      // If neither found, the error will be caught when the symbol is used
    }
    break;
  }

  case ImportKind::All: {
    // from Math import * - import all exports
    // Find all symbols that start with module_name$$
    const std::string prefix = module_name + "$$";

    // Check module_exports_ for what's exported
    auto exports_it = module_exports_.find(module_name);
    if (exports_it != module_exports_.end()) {
      for (const auto& export_name : exports_it->second) {
        const std::string mangled_name = prefix + export_name;

        // Import variable if exists
        auto type_it = local_types_.find(mangled_name);
        if (type_it != local_types_.end()) {
          local_types_[export_name] = type_it->second;
          imported_symbols_[export_name] = mangled_name;
        }

        // Import function if exists
        auto ret_it = function_return_types_.find(mangled_name);
        if (ret_it != function_return_types_.end()) {
          function_return_types_[export_name] = ret_it->second;
          auto param_it = function_param_types_.find(mangled_name);
          if (param_it != function_param_types_.end()) {
            function_param_types_[export_name] = param_it->second;
          }
          imported_symbols_[export_name] = mangled_name;
        }
      }
    }
    break;
  }
  }
}

std::set<std::string> TypeChecker::collectFreeVariables(
    const NBlock& block, const std::set<std::string>& local_names) const {
  FreeVariableCollector collector(local_names);
  block.accept(collector);
  return collector.getReferencedNonLocals();
}
