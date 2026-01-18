#include "parser/ast_printer.hpp"

#include "parser/node.hpp"
#include "parser/operator_utils.hpp"

ASTPrinter::ASTPrinter(std::ostream& out) noexcept : out(out) {}

void ASTPrinter::print(const NBlock& root) {
  depthHasMore.clear();
  out << "NBlock\n";
  const auto& stmts = root.statements;
  for (size_t i = 0; i < stmts.size(); ++i) {
    const bool isLast = (i == stmts.size() - 1);
    DepthScope scope(*this, !isLast);
    stmts[i]->accept(*this);
  }
}

void ASTPrinter::printPrefix() const {
  for (size_t i = 0; i + 1 < depthHasMore.size(); ++i) {
    out << (depthHasMore[i] ? "| " : "  ");
  }
  if (!depthHasMore.empty()) {
    out << (depthHasMore.back() ? "|-" : "`-");
  }
}

std::string ASTPrinter::operatorToString(int op) noexcept {
  return polang::operatorToString(op);
}

ASTPrinter::DepthScope::DepthScope(ASTPrinter& printer, bool hasMore) noexcept
    : printer(printer) {
  printer.depthHasMore.push_back(hasMore);
}

ASTPrinter::DepthScope::~DepthScope() noexcept {
  printer.depthHasMore.pop_back();
}

void ASTPrinter::visit(const NInteger& node) {
  printPrefix();
  out << "NInteger " << node.value << "\n";
}

void ASTPrinter::visit(const NDouble& node) {
  printPrefix();
  out << "NDouble " << node.value << "\n";
}

void ASTPrinter::visit(const NBoolean& node) {
  printPrefix();
  out << "NBoolean " << (node.value ? "true" : "false") << "\n";
}

void ASTPrinter::visit(const NIdentifier& node) {
  printPrefix();
  out << "NIdentifier '" << node.name << "'\n";
}

void ASTPrinter::visit(const NQualifiedName& node) {
  printPrefix();
  out << "NQualifiedName '" << node.fullName() << "'\n";
}

void ASTPrinter::visit(const NMethodCall& node) {
  printPrefix();
  if (node.qualifiedId != nullptr) {
    out << "NMethodCall '" << node.qualifiedId->fullName() << "'\n";
  } else {
    out << "NMethodCall '" << node.id->name << "'\n";
  }

  const auto& args = node.arguments;
  for (size_t i = 0; i < args.size(); ++i) {
    const bool isLast = (i == args.size() - 1);
    DepthScope scope(*this, !isLast);
    args[i]->accept(*this);
  }
}

void ASTPrinter::visit(const NBinaryOperator& node) {
  printPrefix();
  out << "NBinaryOperator '" << operatorToString(node.op) << "'\n";

  {
    DepthScope scope(*this, true);
    node.lhs->accept(*this);
  }
  {
    DepthScope scope(*this, false);
    node.rhs->accept(*this);
  }
}

void ASTPrinter::visit(const NCastExpression& node) {
  printPrefix();
  out << "NCastExpression -> " << node.targetType->name << "\n";

  {
    DepthScope scope(*this, false);
    node.expression->accept(*this);
  }
}

void ASTPrinter::visit(const NAssignment& node) {
  printPrefix();
  out << "NAssignment\n";

  {
    DepthScope scope(*this, true);
    node.lhs->accept(*this);
  }
  {
    DepthScope scope(*this, false);
    node.rhs->accept(*this);
  }
}

void ASTPrinter::visit(const NBlock& node) {
  printPrefix();
  out << "NBlock\n";

  const auto& stmts = node.statements;
  for (size_t i = 0; i < stmts.size(); ++i) {
    const bool isLast = (i == stmts.size() - 1);
    DepthScope scope(*this, !isLast);
    stmts[i]->accept(*this);
  }
}

void ASTPrinter::visit(const NIfExpression& node) {
  printPrefix();
  out << "NIfExpression\n";

  {
    DepthScope scope(*this, true);
    printPrefix();
    out << "condition:\n";
    {
      DepthScope inner(*this, false);
      node.condition->accept(*this);
    }
  }
  {
    DepthScope scope(*this, true);
    printPrefix();
    out << "then:\n";
    {
      DepthScope inner(*this, false);
      node.thenExpr->accept(*this);
    }
  }
  {
    DepthScope scope(*this, false);
    printPrefix();
    out << "else:\n";
    {
      DepthScope inner(*this, false);
      node.elseExpr->accept(*this);
    }
  }
}

void ASTPrinter::visit(const NLetExpression& node) {
  printPrefix();
  out << "NLetExpression\n";

  for (const auto& binding : node.bindings) {
    DepthScope scope(*this, true);
    if (binding->isFunction) {
      binding->func->accept(*this);
    } else {
      binding->var->accept(*this);
    }
  }
  {
    DepthScope scope(*this, false);
    printPrefix();
    out << "body:\n";
    {
      DepthScope inner(*this, false);
      node.body->accept(*this);
    }
  }
}

void ASTPrinter::visit(const NExpressionStatement& node) {
  printPrefix();
  out << "NExpressionStatement\n";

  {
    DepthScope scope(*this, false);
    node.expression->accept(*this);
  }
}

void ASTPrinter::visit(const NVariableDeclaration& node) {
  printPrefix();
  out << "NVariableDeclaration '" << node.id->name << "'";
  if (node.isMutable) {
    out << " mut";
  }
  if (node.type != nullptr) {
    out << " : " << node.type->name;
  }
  out << "\n";

  if (node.assignmentExpr != nullptr) {
    DepthScope scope(*this, false);
    node.assignmentExpr->accept(*this);
  }
}

void ASTPrinter::visit(const NFunctionDeclaration& node) {
  printPrefix();
  out << "NFunctionDeclaration '" << node.id->name << "' (";

  for (size_t i = 0; i < node.arguments.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << node.arguments[i]->id->name;
    if (node.arguments[i]->type != nullptr) {
      out << ": " << node.arguments[i]->type->name;
    }
  }
  out << ")";

  if (node.type != nullptr) {
    out << " -> " << node.type->name;
  }
  out << "\n";

  {
    DepthScope scope(*this, false);
    node.block->accept(*this);
  }
}

void ASTPrinter::visit(const NModuleDeclaration& node) {
  printPrefix();
  out << "NModuleDeclaration '" << node.name->name << "'";

  // Print export list if present
  if (!node.exports.empty()) {
    out << " (";
    for (size_t i = 0; i < node.exports.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << node.exports[i];
    }
    out << ")";
  }
  out << "\n";

  // Print module members
  for (size_t i = 0; i < node.members.size(); ++i) {
    const bool isLast = (i == node.members.size() - 1);
    DepthScope scope(*this, !isLast);
    node.members[i]->accept(*this);
  }
}

void ASTPrinter::visit(const NImportStatement& node) {
  printPrefix();
  out << "NImportStatement ";

  switch (node.kind) {
  case ImportKind::Module:
    out << "import " << node.modulePath->fullName();
    break;
  case ImportKind::ModuleAlias:
    out << "import " << node.modulePath->fullName() << " as " << node.alias;
    break;
  case ImportKind::Items:
    out << "from " << node.modulePath->fullName() << " import ";
    for (size_t i = 0; i < node.items.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << node.items[i].name;
      if (!node.items[i].alias.empty()) {
        out << " as " << node.items[i].alias;
      }
    }
    break;
  case ImportKind::All:
    out << "from " << node.modulePath->fullName() << " import *";
    break;
  }
  out << "\n";
}
