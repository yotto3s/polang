#include "parser/ast_printer.hpp"

#include "parser/node.hpp"
#include "parser/operator_utils.hpp"

ASTPrinter::ASTPrinter(std::ostream& out) noexcept : out_(out) {}

void ASTPrinter::print(const NBlock& root) {
  depth_has_more_.clear();
  out_ << "NBlock\n";
  const auto& stmts = root.statements;
  for (size_t i = 0; i < stmts.size(); ++i) {
    const bool is_last = (i == stmts.size() - 1);
    DepthScope scope(*this, !is_last);
    stmts[i]->accept(*this);
  }
}

void ASTPrinter::printPrefix() const {
  for (size_t i = 0; i + 1 < depth_has_more_.size(); ++i) {
    out_ << (depth_has_more_[i] ? "| " : "  ");
  }
  if (!depth_has_more_.empty()) {
    out_ << (depth_has_more_.back() ? "|-" : "`-");
  }
}

std::string ASTPrinter::operatorToString(int op) noexcept {
  return polang::operatorToString(op);
}

ASTPrinter::DepthScope::DepthScope(ASTPrinter& printer, bool has_more) noexcept
    : printer_(printer) {
  printer_.depth_has_more_.push_back(has_more);
}

ASTPrinter::DepthScope::~DepthScope() noexcept {
  printer_.depth_has_more_.pop_back();
}

void ASTPrinter::visit(const NInteger& node) {
  printPrefix();
  out_ << "NInteger " << node.value << "\n";
}

void ASTPrinter::visit(const NDouble& node) {
  printPrefix();
  out_ << "NDouble " << node.value << "\n";
}

void ASTPrinter::visit(const NBoolean& node) {
  printPrefix();
  out_ << "NBoolean " << (node.value ? "true" : "false") << "\n";
}

void ASTPrinter::visit(const NIdentifier& node) {
  printPrefix();
  out_ << "NIdentifier '" << node.name << "'\n";
}

void ASTPrinter::visit(const NQualifiedName& node) {
  printPrefix();
  out_ << "NQualifiedName '" << node.fullName() << "'\n";
}

void ASTPrinter::visit(const NMethodCall& node) {
  printPrefix();
  if (node.qualifiedId) {
    out_ << "NMethodCall '" << node.qualifiedId->fullName() << "'\n";
  } else {
    out_ << "NMethodCall '" << node.id.name << "'\n";
  }

  const auto& args = node.arguments;
  for (size_t i = 0; i < args.size(); ++i) {
    const bool is_last = (i == args.size() - 1);
    DepthScope scope(*this, !is_last);
    args[i]->accept(*this);
  }
}

void ASTPrinter::visit(const NBinaryOperator& node) {
  printPrefix();
  out_ << "NBinaryOperator '" << operatorToString(node.op) << "'\n";

  {
    DepthScope scope(*this, true);
    node.lhs.accept(*this);
  }
  {
    DepthScope scope(*this, false);
    node.rhs.accept(*this);
  }
}

void ASTPrinter::visit(const NAssignment& node) {
  printPrefix();
  out_ << "NAssignment\n";

  {
    DepthScope scope(*this, true);
    node.lhs.accept(*this);
  }
  {
    DepthScope scope(*this, false);
    node.rhs.accept(*this);
  }
}

void ASTPrinter::visit(const NBlock& node) {
  printPrefix();
  out_ << "NBlock\n";

  const auto& stmts = node.statements;
  for (size_t i = 0; i < stmts.size(); ++i) {
    const bool is_last = (i == stmts.size() - 1);
    DepthScope scope(*this, !is_last);
    stmts[i]->accept(*this);
  }
}

void ASTPrinter::visit(const NIfExpression& node) {
  printPrefix();
  out_ << "NIfExpression\n";

  {
    DepthScope scope(*this, true);
    printPrefix();
    out_ << "condition:\n";
    {
      DepthScope inner(*this, false);
      node.condition.accept(*this);
    }
  }
  {
    DepthScope scope(*this, true);
    printPrefix();
    out_ << "then:\n";
    {
      DepthScope inner(*this, false);
      node.thenExpr.accept(*this);
    }
  }
  {
    DepthScope scope(*this, false);
    printPrefix();
    out_ << "else:\n";
    {
      DepthScope inner(*this, false);
      node.elseExpr.accept(*this);
    }
  }
}

void ASTPrinter::visit(const NLetExpression& node) {
  printPrefix();
  out_ << "NLetExpression\n";

  const auto& bindings = node.bindings;
  for (size_t i = 0; i < bindings.size(); ++i) {
    DepthScope scope(*this, true);
    if (bindings[i]->isFunction) {
      bindings[i]->func->accept(*this);
    } else {
      bindings[i]->var->accept(*this);
    }
  }
  {
    DepthScope scope(*this, false);
    printPrefix();
    out_ << "body:\n";
    {
      DepthScope inner(*this, false);
      node.body.accept(*this);
    }
  }
}

void ASTPrinter::visit(const NExpressionStatement& node) {
  printPrefix();
  out_ << "NExpressionStatement\n";

  {
    DepthScope scope(*this, false);
    node.expression.accept(*this);
  }
}

void ASTPrinter::visit(const NVariableDeclaration& node) {
  printPrefix();
  out_ << "NVariableDeclaration '" << node.id.name << "'";
  if (node.isMutable) {
    out_ << " mut";
  }
  if (node.type != nullptr) {
    out_ << " : " << node.type->name;
  }
  out_ << "\n";

  if (node.assignmentExpr != nullptr) {
    DepthScope scope(*this, false);
    node.assignmentExpr->accept(*this);
  }
}

void ASTPrinter::visit(const NFunctionDeclaration& node) {
  printPrefix();
  out_ << "NFunctionDeclaration '" << node.id.name << "' (";

  for (size_t i = 0; i < node.arguments.size(); ++i) {
    if (i > 0) {
      out_ << ", ";
    }
    out_ << node.arguments[i]->id.name;
    if (node.arguments[i]->type != nullptr) {
      out_ << ": " << node.arguments[i]->type->name;
    }
  }
  out_ << ")";

  if (node.type != nullptr) {
    out_ << " -> " << node.type->name;
  }
  out_ << "\n";

  {
    DepthScope scope(*this, false);
    node.block.accept(*this);
  }
}

void ASTPrinter::visit(const NModuleDeclaration& node) {
  printPrefix();
  out_ << "NModuleDeclaration '" << node.name.name << "'";

  // Print export list if present
  if (!node.exports.empty()) {
    out_ << " (";
    for (size_t i = 0; i < node.exports.size(); ++i) {
      if (i > 0) {
        out_ << ", ";
      }
      out_ << node.exports[i];
    }
    out_ << ")";
  }
  out_ << "\n";

  // Print module members
  for (size_t i = 0; i < node.members.size(); ++i) {
    const bool is_last = (i == node.members.size() - 1);
    DepthScope scope(*this, !is_last);
    node.members[i]->accept(*this);
  }
}

void ASTPrinter::visit(const NImportStatement& node) {
  printPrefix();
  out_ << "NImportStatement ";

  switch (node.kind) {
  case ImportKind::Module:
    out_ << "import " << node.modulePath.fullName();
    break;
  case ImportKind::ModuleAlias:
    out_ << "import " << node.modulePath.fullName() << " as " << node.alias;
    break;
  case ImportKind::Items:
    out_ << "from " << node.modulePath.fullName() << " import ";
    for (size_t i = 0; i < node.items.size(); ++i) {
      if (i > 0) {
        out_ << ", ";
      }
      out_ << node.items[i].name;
      if (!node.items[i].alias.empty()) {
        out_ << " as " << node.items[i].alias;
      }
    }
    break;
  case ImportKind::All:
    out_ << "from " << node.modulePath.fullName() << " import *";
    break;
  }
  out_ << "\n";
}
