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

/// ParameterTypeInferrer - A visitor that infers types of untyped function
/// parameters by analyzing how they are used in the function body.
///
/// ## Algorithm
///
/// 1. Track a set of parameters whose types need to be inferred
/// 2. Traverse the function body and collect type constraints:
///    - Binary operations: if `param + literal`, infer param type from literal
///    - Binary operations: if `param + var`, infer param type from var's type
///    - Function calls: if `f(param)`, infer param type from f's parameter type
///    - If condition: if `if param then ...`, infer param is bool
/// 3. Return the collected type constraints
class ParameterTypeInferrer : public Visitor {
public:
  ParameterTypeInferrer(
      const std::set<std::string>& untyped_params,
      const std::map<std::string, std::string>& known_types,
      const std::map<std::string, std::vector<std::string>>& func_param_types,
      const std::map<std::string, std::string>& func_return_types)
      : untyped_params_(untyped_params), known_types_(known_types),
        func_param_types_(func_param_types),
        func_return_types_(func_return_types) {}

  std::map<std::string, std::string> getInferredTypes() const {
    return inferred_types_;
  }

  void visit(const NInteger& node) override { current_type_ = TypeNames::INT; }

  void visit(const NDouble& node) override {
    current_type_ = TypeNames::DOUBLE;
  }

  void visit(const NBoolean& node) override { current_type_ = TypeNames::BOOL; }

  void visit(const NIdentifier& node) override {
    // Check if this is an untyped parameter
    if (untyped_params_.count(node.name)) {
      // If we already inferred a type, use it
      if (inferred_types_.count(node.name)) {
        current_type_ = inferred_types_[node.name];
      } else {
        current_type_ = TypeNames::UNKNOWN;
      }
    } else if (known_types_.count(node.name)) {
      current_type_ = known_types_.at(node.name);
    } else {
      current_type_ = TypeNames::UNKNOWN;
    }
  }

  void visit(const NBinaryOperator& node) override {
    // Get types of both operands
    node.lhs.accept(*this);
    const std::string lhs_type = current_type_;
    const std::string lhs_name = getIdentifierName(node.lhs);

    node.rhs.accept(*this);
    const std::string rhs_type = current_type_;
    const std::string rhs_name = getIdentifierName(node.rhs);

    // If one side is an untyped param and the other has a known type,
    // infer the param's type
    if (!lhs_name.empty() && untyped_params_.count(lhs_name) &&
        rhs_type != TypeNames::UNKNOWN) {
      addConstraint(lhs_name, rhs_type);
    }
    if (!rhs_name.empty() && untyped_params_.count(rhs_name) &&
        lhs_type != TypeNames::UNKNOWN) {
      addConstraint(rhs_name, lhs_type);
    }

    // If both sides are untyped params used together, they must have same type
    // but we can't infer what type yet

    // Determine result type
    if (isComparisonOperator(node.op)) {
      current_type_ = TypeNames::BOOL;
    } else if (lhs_type != TypeNames::UNKNOWN) {
      current_type_ = lhs_type;
    } else if (rhs_type != TypeNames::UNKNOWN) {
      current_type_ = rhs_type;
    } else {
      current_type_ = TypeNames::UNKNOWN;
    }
  }

  void visit(const NMethodCall& node) override {
    // Check if we know the function's parameter types
    auto func_it = func_param_types_.find(node.id.name);
    if (func_it != func_param_types_.end()) {
      const auto& param_types = func_it->second;
      for (size_t i = 0; i < node.arguments.size() && i < param_types.size();
           ++i) {
        const std::string param_name = getIdentifierName(*node.arguments[i]);
        if (!param_name.empty() && untyped_params_.count(param_name) &&
            param_types[i] != TypeNames::UNKNOWN) {
          addConstraint(param_name, param_types[i]);
        }
        // Also recurse into arguments
        node.arguments[i]->accept(*this);
      }
    } else {
      // Unknown function - just recurse
      for (const auto* arg : node.arguments) {
        arg->accept(*this);
      }
    }

    // Set return type
    auto ret_it = func_return_types_.find(node.id.name);
    if (ret_it != func_return_types_.end()) {
      current_type_ = ret_it->second;
    } else {
      current_type_ = TypeNames::UNKNOWN;
    }
  }

  void visit(const NAssignment& node) override {
    // Visit RHS
    node.rhs.accept(*this);
    const std::string rhs_type = current_type_;

    // If LHS is an untyped parameter and RHS has known type
    if (untyped_params_.count(node.lhs.name) &&
        rhs_type != TypeNames::UNKNOWN) {
      addConstraint(node.lhs.name, rhs_type);
    }

    current_type_ = rhs_type;
  }

  void visit(const NBlock& node) override {
    for (const auto* stmt : node.statements) {
      stmt->accept(*this);
    }
  }

  void visit(const NIfExpression& node) override {
    // If condition is an untyped param, it must be bool
    const std::string cond_name = getIdentifierName(node.condition);
    if (!cond_name.empty() && untyped_params_.count(cond_name)) {
      addConstraint(cond_name, TypeNames::BOOL);
    }

    // Recurse into condition
    node.condition.accept(*this);

    // Recurse into branches
    node.thenExpr.accept(*this);
    const std::string then_type = current_type_;
    node.elseExpr.accept(*this);

    current_type_ = then_type;
  }

  void visit(const NLetExpression& node) override {
    // Save and extend known types with let bindings
    auto saved_known_types = known_types_;

    // First pass: process initializers
    for (const auto* binding : node.bindings) {
      if (!binding->isFunction && binding->var->assignmentExpr != nullptr) {
        binding->var->assignmentExpr->accept(*this);
        if (current_type_ != TypeNames::UNKNOWN) {
          known_types_[binding->var->id.name] = current_type_;
        }
      }
    }

    // Add function bindings to scope (for recursive calls)
    for (const auto* binding : node.bindings) {
      if (binding->isFunction) {
        // Note: we don't recurse into nested function bodies
        known_types_[binding->func->id.name] = TypeNames::FUNCTION;
      }
    }

    // Process body
    node.body.accept(*this);

    // Restore
    known_types_ = saved_known_types;
  }

  void visit(const NExpressionStatement& node) override {
    node.expression.accept(*this);
  }

  void visit(const NVariableDeclaration& node) override {
    if (node.assignmentExpr != nullptr) {
      node.assignmentExpr->accept(*this);
    }
    // Add to known types for subsequent traversal
    if (node.type != nullptr) {
      known_types_[node.id.name] = node.type->name;
    } else if (current_type_ != TypeNames::UNKNOWN) {
      known_types_[node.id.name] = current_type_;
    }
  }

  void visit(const NFunctionDeclaration& node) override {
    // Don't recurse into nested function bodies
    known_types_[node.id.name] = TypeNames::FUNCTION;
  }

private:
  std::set<std::string> untyped_params_;
  std::map<std::string, std::string> known_types_;
  std::map<std::string, std::vector<std::string>> func_param_types_;
  std::map<std::string, std::string> func_return_types_;
  std::map<std::string, std::string> inferred_types_;
  std::string current_type_;

  void addConstraint(const std::string& param, const std::string& type) {
    if (inferred_types_.count(param) == 0) {
      inferred_types_[param] = type;
    }
    // If already has a type, could check for conflict, but for now just keep
    // first
  }

  /// Get the identifier name if the expression is a simple identifier
  std::string getIdentifierName(const NExpression& expr) const {
    // Try to cast to NIdentifier
    const auto* ident = dynamic_cast<const NIdentifier*>(&expr);
    if (ident != nullptr) {
      return ident->name;
    }
    return "";
  }
};

/// CallSiteCollector - A visitor that collects call sites to specific functions
/// and their argument types. Used for call-site type inference in
/// let-expressions.
class CallSiteCollector : public Visitor {
public:
  CallSiteCollector(const std::set<std::string>& tracked_functions,
                    const std::map<std::string, std::string>& known_types)
      : tracked_functions_(tracked_functions), known_types_(known_types) {}

  /// Get collected call sites: function name -> list of (arg_position -> type)
  const std::map<std::string, std::vector<std::vector<std::string>>>&
  getCallSites() const {
    return call_sites_;
  }

  void visit(const NInteger& node) override { current_type_ = TypeNames::INT; }

  void visit(const NDouble& node) override {
    current_type_ = TypeNames::DOUBLE;
  }

  void visit(const NBoolean& node) override { current_type_ = TypeNames::BOOL; }

  void visit(const NIdentifier& node) override {
    if (known_types_.count(node.name)) {
      current_type_ = known_types_.at(node.name);
    } else {
      current_type_ = TypeNames::UNKNOWN;
    }
  }

  void visit(const NMethodCall& node) override {
    // Collect argument types
    std::vector<std::string> arg_types;
    for (const auto* arg : node.arguments) {
      arg->accept(*this);
      arg_types.push_back(current_type_);
    }

    // If this is a tracked function, record the call site
    if (tracked_functions_.count(node.id.name)) {
      call_sites_[node.id.name].push_back(arg_types);
    }

    current_type_ = TypeNames::UNKNOWN; // Don't know return type yet
  }

  void visit(const NBinaryOperator& node) override {
    node.lhs.accept(*this);
    const std::string lhs_type = current_type_;
    node.rhs.accept(*this);
    const std::string rhs_type = current_type_;

    if (isComparisonOperator(node.op)) {
      current_type_ = TypeNames::BOOL;
    } else if (lhs_type != TypeNames::UNKNOWN) {
      current_type_ = lhs_type;
    } else {
      current_type_ = rhs_type;
    }
  }

  void visit(const NAssignment& node) override { node.rhs.accept(*this); }

  void visit(const NBlock& node) override {
    for (const auto* stmt : node.statements) {
      stmt->accept(*this);
    }
  }

  void visit(const NIfExpression& node) override {
    node.condition.accept(*this);
    node.thenExpr.accept(*this);
    const std::string then_type = current_type_;
    node.elseExpr.accept(*this);
    current_type_ = then_type;
  }

  void visit(const NLetExpression& node) override {
    // Extend known types with let bindings
    auto saved_known_types = known_types_;

    for (const auto* binding : node.bindings) {
      if (!binding->isFunction && binding->var->assignmentExpr != nullptr) {
        binding->var->assignmentExpr->accept(*this);
        if (binding->var->type != nullptr) {
          known_types_[binding->var->id.name] = binding->var->type->name;
        } else if (current_type_ != TypeNames::UNKNOWN) {
          known_types_[binding->var->id.name] = current_type_;
        }
      }
    }

    node.body.accept(*this);
    known_types_ = saved_known_types;
  }

  void visit(const NExpressionStatement& node) override {
    node.expression.accept(*this);
  }

  void visit(const NVariableDeclaration& node) override {
    if (node.assignmentExpr != nullptr) {
      node.assignmentExpr->accept(*this);
    }
    if (node.type != nullptr) {
      known_types_[node.id.name] = node.type->name;
    }
  }

  void visit(const NFunctionDeclaration& node) override {
    // Don't recurse into nested function bodies
  }

private:
  std::set<std::string> tracked_functions_;
  std::map<std::string, std::string> known_types_;
  std::map<std::string, std::vector<std::vector<std::string>>> call_sites_;
  std::string current_type_;
};

TypeChecker::TypeChecker() : inferred_type_(TypeNames::INT) {}

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
        if (arg_types[i] != TypeNames::UNKNOWN &&
            param_types[i] != TypeNames::UNKNOWN &&
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

  if (isArithmeticOperator(node.op)) {
    // Both operands must be the same type for arithmetic
    if (lhs_type != rhs_type) {
      reportError("Type mismatch in '" + operatorToString(node.op) +
                  "': " + lhs_type + " and " + rhs_type);
    }
    inferred_type_ = lhs_type; // Result has same type as operands
  } else if (isComparisonOperator(node.op)) {
    // Comparisons also require same types
    if (lhs_type != rhs_type) {
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

  if (rhs_type != TypeNames::UNKNOWN && rhs_type != var_type) {
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
  if (cond_type != TypeNames::UNKNOWN && cond_type != TypeNames::BOOL) {
    reportError("If condition must be bool, got " + cond_type);
  }

  // Check then branch
  node.thenExpr.accept(*this);
  const std::string then_type = inferred_type_;

  // Check else branch
  node.elseExpr.accept(*this);
  const std::string else_type = inferred_type_;

  // Both branches must have same type
  if (then_type != TypeNames::UNKNOWN && else_type != TypeNames::UNKNOWN &&
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

  // Pass 0: Identify functions with untyped parameters for call-site inference
  std::set<std::string> funcs_needing_inference;
  for (const auto* binding : node.bindings) {
    if (binding->isFunction) {
      for (const auto* arg : binding->func->arguments) {
        if (arg->type == nullptr) {
          funcs_needing_inference.insert(binding->func->id.name);
          break;
        }
      }
    }
  }

  // Pass 0.5: Collect call sites from the body for functions needing inference
  std::map<std::string, std::vector<std::vector<std::string>>>
      call_site_arg_types;
  if (!funcs_needing_inference.empty()) {
    CallSiteCollector collector(funcs_needing_inference, saved_locals);
    node.body.accept(collector);
    call_site_arg_types = collector.getCallSites();
  }

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

      // Collect parameter names and identify untyped parameters
      std::set<std::string> param_names;
      std::set<std::string> untyped_params;
      std::vector<std::string> param_types;
      for (const auto* arg : func->arguments) {
        param_names.insert(arg->id.name);
        if (arg->type == nullptr) {
          untyped_params.insert(arg->id.name);
          param_types.push_back(TypeNames::UNKNOWN);
        } else {
          local_types_[arg->id.name] = arg->type->name;
          local_mutability_[arg->id.name] = arg->isMutable;
          param_types.push_back(arg->type->name);
        }
      }

      // If there are untyped parameters, run type inference
      if (!untyped_params.empty()) {
        // First try body inference
        ParameterTypeInferrer body_inferrer(untyped_params, local_types_,
                                            function_param_types_,
                                            function_return_types_);
        func->block.accept(body_inferrer);
        auto inferred = body_inferrer.getInferredTypes();

        // Then try call-site inference
        auto call_it = call_site_arg_types.find(func->id.name);
        if (call_it != call_site_arg_types.end()) {
          for (const auto& call_args : call_it->second) {
            size_t arg_idx = 0;
            for (const auto* arg : func->arguments) {
              if (arg_idx < call_args.size() && arg->type == nullptr &&
                  inferred.count(arg->id.name) == 0 &&
                  call_args[arg_idx] != TypeNames::UNKNOWN) {
                inferred[arg->id.name] = call_args[arg_idx];
              }
              ++arg_idx;
            }
          }
        }

        // Update parameter types in AST and local scope
        size_t idx = 0;
        for (const auto* arg : func->arguments) {
          if (arg->type == nullptr) {
            auto& mutable_arg = const_cast<NVariableDeclaration&>(*arg);
            auto it = inferred.find(arg->id.name);
            if (it != inferred.end() && it->second != TypeNames::UNKNOWN) {
              mutable_arg.type = new NIdentifier(it->second);
              local_types_[arg->id.name] = it->second;
              local_mutability_[arg->id.name] = arg->isMutable;
              param_types[idx] = it->second;
            } else {
              reportError("Cannot infer type of parameter '" + arg->id.name +
                          "' in function '" + func->id.name + "'");
            }
          }
          ++idx;
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

      // Handle return type inference
      if (func->type == nullptr) {
        if (body_type != TypeNames::UNKNOWN) {
          mutable_func.type = new NIdentifier(body_type);
        } else {
          mutable_func.type = new NIdentifier(TypeNames::INT);
        }
      } else if (body_type != TypeNames::UNKNOWN &&
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

  // Pass 2: Add all bindings to scope
  std::size_t i = 0;
  std::size_t func_idx = 0;
  for (const auto* binding : node.bindings) {
    if (binding->isFunction) {
      const auto* func = binding->func;
      function_param_types_[func->id.name] = func_param_types[func_idx++];
      function_return_types_[func->id.name] =
          func->type ? func->type->name : TypeNames::INT;
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
      inferred_type_ = TypeNames::UNKNOWN;
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

  if (expr_type == TypeNames::UNKNOWN) {
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

  // Collect parameter names and identify untyped parameters
  std::set<std::string> param_names;
  std::set<std::string> untyped_params;
  std::vector<std::string> param_types;
  for (const auto* arg : node.arguments) {
    param_names.insert(arg->id.name);
    if (arg->type == nullptr) {
      untyped_params.insert(arg->id.name);
      param_types.push_back(TypeNames::UNKNOWN);
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

  // If there are untyped parameters, run type inference on the body
  if (!untyped_params.empty()) {
    ParameterTypeInferrer inferrer(untyped_params, local_types_,
                                   function_param_types_,
                                   function_return_types_);
    node.block.accept(inferrer);
    const auto inferred = inferrer.getInferredTypes();

    // Update parameter types in AST and local scope
    size_t idx = 0;
    for (const auto* arg : node.arguments) {
      if (arg->type == nullptr) {
        auto& mutable_arg = const_cast<NVariableDeclaration&>(*arg);
        auto it = inferred.find(arg->id.name);
        if (it != inferred.end() && it->second != TypeNames::UNKNOWN) {
          mutable_arg.type = new NIdentifier(it->second);
          local_types_[arg->id.name] = it->second;
          local_mutability_[arg->id.name] = arg->isMutable;
          param_types[idx] = it->second;
        } else {
          reportError("Cannot infer type of parameter '" + arg->id.name +
                      "' in function '" + node.id.name + "'");
          param_types[idx] = TypeNames::UNKNOWN;
        }
      }
      ++idx;
    }
  }

  // Store parameter types for this function
  function_param_types_[node.id.name] = param_types;

  // Check function body
  node.block.accept(*this);
  const std::string body_type = inferred_type_;

  if (node.type == nullptr) {
    // Return type not annotated - infer from body
    if (body_type != TypeNames::UNKNOWN) {
      mutable_node.type = new NIdentifier(body_type);
      function_return_types_[node.id.name] = body_type;
    } else {
      // Default to int if we can't infer
      mutable_node.type = new NIdentifier(TypeNames::INT);
      function_return_types_[node.id.name] = TypeNames::INT;
    }
  } else {
    // Return type annotated - validate
    const std::string decl_return_type = node.type->name;

    if (body_type != TypeNames::UNKNOWN && body_type != decl_return_type) {
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
