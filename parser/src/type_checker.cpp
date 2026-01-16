// clang-format off
#include <parser/type_checker.hpp>
#include <parser/node.hpp>
#include "parser.hpp"  // Must be after node.hpp for token constants
// clang-format on

#include <iostream>
#include <set>

// Helper visitor class to collect free variables (identifiers not locally
// defined)
class FreeVariableCollector : public Visitor {
public:
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

private:
  std::set<std::string> local_names_;
  std::set<std::string> referenced_non_locals_;
};

TypeChecker::TypeChecker() : inferred_type_("int") {}

std::vector<TypeCheckError> TypeChecker::check(const NBlock& ast) {
  errors_.clear();
  local_types_.clear();
  local_mutability_.clear();
  function_return_types_.clear();
  function_param_types_.clear();
  ast.accept(*this);
  return errors_;
}

std::string TypeChecker::inferType(const Node& node) {
  node.accept(*this);
  return inferred_type_;
}

void TypeChecker::reportError(const std::string& message) {
  errors_.push_back(TypeCheckError(message));
  std::cerr << "Type error: " << message << std::endl;
}

void TypeChecker::visit(const NInteger& node) { inferred_type_ = "int"; }

void TypeChecker::visit(const NDouble& node) { inferred_type_ = "double"; }

void TypeChecker::visit(const NBoolean& node) { inferred_type_ = "bool"; }

void TypeChecker::visit(const NIdentifier& node) {
  if (local_types_.find(node.name) == local_types_.end()) {
    reportError("Undeclared variable: " + node.name);
    inferred_type_ = "unknown";
    return;
  }
  inferred_type_ = local_types_[node.name];
}

void TypeChecker::visit(const NMethodCall& node) {
  // Collect argument types
  std::vector<std::string> arg_types;
  for (const auto* arg : node.arguments) {
    arg->accept(*this);
    arg_types.push_back(inferred_type_);
  }

  // Check if function is known
  const auto param_it = function_param_types_.find(node.id.name);
  if (param_it != function_param_types_.end()) {
    const auto& param_types = param_it->second;

    // Check argument count
    if (arg_types.size() != param_types.size()) {
      reportError("Function '" + node.id.name + "' expects " +
                  std::to_string(param_types.size()) + " argument(s), got " +
                  std::to_string(arg_types.size()));
    } else {
      // Check each argument type
      for (std::size_t i = 0; i < arg_types.size(); ++i) {
        if (arg_types[i] != "unknown" && param_types[i] != "unknown" &&
            arg_types[i] != param_types[i]) {
          reportError("Function '" + node.id.name + "' argument " +
                      std::to_string(i + 1) + " expects " + param_types[i] +
                      ", got " + arg_types[i]);
        }
      }
    }
  }

  // Get return type from function
  if (function_return_types_.find(node.id.name) !=
      function_return_types_.end()) {
    inferred_type_ = function_return_types_[node.id.name];
  } else {
    // Unknown function, assume int
    inferred_type_ = "int";
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
  if (lhs_type == "unknown" || rhs_type == "unknown") {
    inferred_type_ = "unknown";
    return;
  }

  const bool is_arithmetic = (node.op == TPLUS || node.op == TMINUS ||
                              node.op == TMUL || node.op == TDIV);
  const bool is_comparison =
      (node.op == TCEQ || node.op == TCNE || node.op == TCLT ||
       node.op == TCLE || node.op == TCGT || node.op == TCGE);

  if (is_arithmetic) {
    // Both operands must be the same type for arithmetic
    if (lhs_type != rhs_type) {
      std::string op_name;
      switch (node.op) {
      case TPLUS:
        op_name = "+";
        break;
      case TMINUS:
        op_name = "-";
        break;
      case TMUL:
        op_name = "*";
        break;
      case TDIV:
        op_name = "/";
        break;
      default:
        op_name = "?";
        break;
      }
      reportError("Type mismatch in '" + op_name + "': " + lhs_type + " and " +
                  rhs_type);
    }
    inferred_type_ = lhs_type; // Result has same type as operands
  } else if (is_comparison) {
    // Comparisons also require same types
    if (lhs_type != rhs_type) {
      reportError("Type mismatch in comparison: " + lhs_type + " and " +
                  rhs_type);
    }
    // Comparison returns bool
    inferred_type_ = "bool";
  }
}

void TypeChecker::visit(const NAssignment& node) {
  // Check variable exists
  if (local_types_.find(node.lhs.name) == local_types_.end()) {
    reportError("Undeclared variable: " + node.lhs.name);
    inferred_type_ = "unknown";
    return;
  }

  // Check mutability
  if (!local_mutability_[node.lhs.name]) {
    reportError("Cannot reassign immutable variable: " + node.lhs.name);
    inferred_type_ = "unknown";
    return;
  }

  const std::string var_type = local_types_[node.lhs.name];

  // Check RHS type
  node.rhs.accept(*this);
  const std::string rhs_type = inferred_type_;

  if (rhs_type != "unknown" && rhs_type != var_type) {
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

  // Condition must be bool
  if (cond_type != "unknown" && cond_type != "bool") {
    reportError("If condition must be bool, got " + cond_type);
  }

  // Check then branch
  node.thenExpr.accept(*this);
  const std::string then_type = inferred_type_;

  // Check else branch
  node.elseExpr.accept(*this);
  const std::string else_type = inferred_type_;

  // Both branches must have same type
  if (then_type != "unknown" && else_type != "unknown" &&
      then_type != else_type) {
    reportError("If branches have different types: " + then_type + " and " +
                else_type);
  }

  inferred_type_ = then_type;
}

void TypeChecker::visit(const NLetExpression& node) {
  // Save current scopes
  const auto saved_locals = local_types_;
  const auto saved_mutability = local_mutability_;
  const auto saved_func_returns = function_return_types_;
  const auto saved_func_params = function_param_types_;

  // Pass 1: First collect sibling variable binding types (for closure capture)
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
        if (inferred_type_ != "unknown") {
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

      // Collect parameter names and add to scope for type-checking the body
      std::set<std::string> param_names;
      std::vector<std::string> param_types;
      for (const auto* arg : func->arguments) {
        param_names.insert(arg->id.name);
        if (arg->type == nullptr) {
          reportError("Function parameter '" + arg->id.name +
                      "' must have type annotation");
          param_types.push_back("unknown");
          continue;
        }
        local_types_[arg->id.name] = arg->type->name;
        local_mutability_[arg->id.name] = arg->isMutable;
        param_types.push_back(arg->type->name);
      }
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

      // Handle return type inference
      if (func->type == nullptr) {
        if (body_type != "unknown") {
          mutable_func.type = new NIdentifier(body_type);
        } else {
          mutable_func.type = new NIdentifier("int");
        }
      } else if (body_type != "unknown" && body_type != func->type->name) {
        reportError("Function '" + func->id.name + "' declared to return " +
                    func->type->name + " but body has type " + body_type);
      }

      // Placeholder for function binding type and mutability
      binding_types.push_back("function");
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
          binding_types.push_back("unknown");
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

      if (expr_type == "unknown") {
        if (var->type != nullptr) {
          binding_types.push_back(var->type->name);
          binding_mutability.push_back(var->isMutable);
        } else {
          binding_types.push_back("unknown");
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

  // Pass 2: Add all bindings to scope
  std::size_t i = 0;
  std::size_t func_idx = 0;
  for (const auto* binding : node.bindings) {
    if (binding->isFunction) {
      const auto* func = binding->func;
      function_param_types_[func->id.name] = func_param_types[func_idx++];
      function_return_types_[func->id.name] =
          func->type ? func->type->name : "int";
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

  if (node.assignmentExpr == nullptr) {
    // No initializer - must have type annotation
    if (node.type == nullptr) {
      reportError("Variable '" + node.id.name +
                  "' must have type annotation or initializer");
      inferred_type_ = "unknown";
      return;
    }
    local_types_[node.id.name] = node.type->name;
    local_mutability_[node.id.name] = node.isMutable;
    inferred_type_ = node.type->name;
    return;
  }

  // Infer expression type
  node.assignmentExpr->accept(*this);
  const std::string expr_type = inferred_type_;

  if (expr_type == "unknown") {
    // Error already reported for expression
    if (node.type != nullptr) {
      local_types_[node.id.name] = node.type->name;
      local_mutability_[node.id.name] = node.isMutable;
      inferred_type_ = node.type->name;
    }
    return;
  }

  if (node.type == nullptr) {
    // No type annotation - infer from expression
    mutable_node.type = new NIdentifier(expr_type);
    local_types_[node.id.name] = expr_type;
    local_mutability_[node.id.name] = node.isMutable;
    inferred_type_ = expr_type;
  } else {
    // Type annotation present - validate (no coercion!)
    const std::string decl_type = node.type->name;

    if (expr_type != decl_type) {
      reportError("Variable '" + node.id.name + "' declared as " + decl_type +
                  " but initialized with " + expr_type +
                  " (no implicit conversion)");
    }

    local_types_[node.id.name] = decl_type;
    local_mutability_[node.id.name] = node.isMutable;
    inferred_type_ = decl_type;
  }
}

void TypeChecker::visit(const NFunctionDeclaration& node) {
  // Cast away const to allow setting inferred type
  NFunctionDeclaration& mutable_node = const_cast<NFunctionDeclaration&>(node);

  // Save current scope
  const auto saved_locals = local_types_;
  const auto saved_mutability = local_mutability_;

  // Collect parameter names (these are local to the function)
  std::set<std::string> param_names;
  std::vector<std::string> param_types;
  for (const auto* arg : node.arguments) {
    param_names.insert(arg->id.name);
    if (arg->type == nullptr) {
      reportError("Function parameter '" + arg->id.name +
                  "' must have type annotation");
      param_types.push_back("unknown");
      continue;
    }
    local_types_[arg->id.name] = arg->type->name;
    local_mutability_[arg->id.name] = arg->isMutable;
    param_types.push_back(arg->type->name);
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

  // Store parameter types for this function
  function_param_types_[node.id.name] = param_types;

  // Check function body
  node.block.accept(*this);
  const std::string body_type = inferred_type_;

  if (node.type == nullptr) {
    // Return type not annotated - infer from body
    if (body_type != "unknown") {
      mutable_node.type = new NIdentifier(body_type);
      function_return_types_[node.id.name] = body_type;
    } else {
      // Default to int if we can't infer
      mutable_node.type = new NIdentifier("int");
      function_return_types_[node.id.name] = "int";
    }
  } else {
    // Return type annotated - validate
    const std::string decl_return_type = node.type->name;

    if (body_type != "unknown" && body_type != decl_return_type) {
      reportError("Function '" + node.id.name + "' declared to return " +
                  decl_return_type + " but body has type " + body_type +
                  " (no implicit conversion)");
    }

    function_return_types_[node.id.name] = decl_return_type;
  }

  // Restore previous scope
  local_types_ = saved_locals;
  local_mutability_ = saved_mutability;
  inferred_type_ = node.type ? node.type->name : body_type;
}

std::set<std::string> TypeChecker::collectFreeVariables(
    const NBlock& block, const std::set<std::string>& local_names) const {
  FreeVariableCollector collector(local_names);
  block.accept(collector);
  return collector.getReferencedNonLocals();
}
