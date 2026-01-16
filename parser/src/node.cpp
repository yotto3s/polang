#include <parser/node.hpp>
#include <parser/visitor.hpp>

void NInteger::accept(Visitor& visitor) { visitor.visit(*this); }

void NDouble::accept(Visitor& visitor) { visitor.visit(*this); }

void NIdentifier::accept(Visitor& visitor) { visitor.visit(*this); }

void NMethodCall::accept(Visitor& visitor) { visitor.visit(*this); }

void NBinaryOperator::accept(Visitor& visitor) { visitor.visit(*this); }

void NAssignment::accept(Visitor& visitor) { visitor.visit(*this); }

void NBlock::accept(Visitor& visitor) { visitor.visit(*this); }

void NIfExpression::accept(Visitor& visitor) { visitor.visit(*this); }

void NExpressionStatement::accept(Visitor& visitor) { visitor.visit(*this); }

void NVariableDeclaration::accept(Visitor& visitor) { visitor.visit(*this); }

void NFunctionDeclaration::accept(Visitor& visitor) { visitor.visit(*this); }
