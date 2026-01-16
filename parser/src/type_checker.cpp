// clang-format off
#include <parser/type_checker.hpp>
#include <parser/node.hpp>
#include "parser.hpp"  // Must be after node.hpp for token constants
// clang-format on

#include <iostream>

TypeChecker::TypeChecker() : inferred_type_("int") {}

std::vector<TypeCheckError> TypeChecker::check(const NBlock& ast) {
  errors_.clear();
  local_types_.clear();
  function_return_types_.clear();
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
  // Check argument types (for now, just visit them)
  for (const auto* arg : node.arguments) {
    arg->accept(*this);
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
  // Save current scope
  const auto saved_locals = local_types_;

  // Process bindings
  for (const auto* binding : node.bindings) {
    binding->accept(*this);
  }

  // Process body with new variables in scope
  node.body.accept(*this);

  // Restore previous scope
  local_types_ = saved_locals;
}

void TypeChecker::visit(const NExpressionStatement& node) {
  node.expression.accept(*this);
}

void TypeChecker::visit(const NVariableDeclaration& node) {
  const std::string decl_type = node.type.name;

  // Check initializer if present
  if (node.assignmentExpr != nullptr) {
    node.assignmentExpr->accept(*this);
    const std::string expr_type = inferred_type_;

    if (expr_type != "unknown" && expr_type != decl_type) {
      reportError("Variable '" + node.id.name + "' declared as " + decl_type +
                  " but initialized with " + expr_type);
    }
  }

  // Store the variable type in scope
  local_types_[node.id.name] = decl_type;
  inferred_type_ = decl_type;
}

void TypeChecker::visit(const NFunctionDeclaration& node) {
  // Save current scope
  const auto saved_locals = local_types_;

  // Store function return type
  function_return_types_[node.id.name] = node.type.name;

  // Add parameters to scope
  for (const auto* arg : node.arguments) {
    local_types_[arg->id.name] = arg->type.name;
  }

  // Check function body
  node.block.accept(*this);

  // Check return type matches body expression type
  if (inferred_type_ != "unknown" && inferred_type_ != node.type.name) {
    reportError("Function '" + node.id.name + "' returns " + node.type.name +
                " but body has type " + inferred_type_);
  }

  // Restore previous scope
  local_types_ = saved_locals;
  inferred_type_ = node.type.name;
}
