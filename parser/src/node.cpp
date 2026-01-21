#include <parser/node.hpp>
#include <parser/visitor.hpp>

void NInteger::accept(Visitor& visitor) const { visitor.visit(*this); }

void NDouble::accept(Visitor& visitor) const { visitor.visit(*this); }

void NBoolean::accept(Visitor& visitor) const { visitor.visit(*this); }

void NIdentifier::accept(Visitor& visitor) const { visitor.visit(*this); }

void NNamedType::accept(Visitor& visitor) const { visitor.visit(*this); }

void NQualifiedName::accept(Visitor& visitor) const { visitor.visit(*this); }

void NMethodCall::accept(Visitor& visitor) const { visitor.visit(*this); }

void NBinaryOperator::accept(Visitor& visitor) const { visitor.visit(*this); }

void NCastExpression::accept(Visitor& visitor) const { visitor.visit(*this); }

void NBlock::accept(Visitor& visitor) const { visitor.visit(*this); }

void NIfExpression::accept(Visitor& visitor) const { visitor.visit(*this); }

void NLetExpression::accept(Visitor& visitor) const { visitor.visit(*this); }

void NExpressionStatement::accept(Visitor& visitor) const {
  visitor.visit(*this);
}

void NVariableDeclaration::accept(Visitor& visitor) const {
  visitor.visit(*this);
}

void NFunctionDeclaration::accept(Visitor& visitor) const {
  visitor.visit(*this);
}

void NModuleDeclaration::accept(Visitor& visitor) const {
  visitor.visit(*this);
}

void NImportStatement::accept(Visitor& visitor) const { visitor.visit(*this); }
