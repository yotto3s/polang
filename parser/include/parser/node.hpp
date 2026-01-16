#ifndef POLANG_NODE_HPP
#define POLANG_NODE_HPP

#include <iostream>
#include <string>
#include <vector>

class Visitor;
class NStatement;
class NExpression;
class NVariableDeclaration;

typedef std::vector<NStatement*> StatementList;
typedef std::vector<NExpression*> ExpressionList;
typedef std::vector<NVariableDeclaration*> VariableList;

// clang-format off
class Node {
public:
  virtual ~Node() {}
  virtual void accept(Visitor &visitor) const = 0;
};

class NExpression : public Node {};

class NStatement : public Node {};

class NInteger : public NExpression {
public:
  long long value;
  NInteger(long long value) : value(value) {}
  void accept(Visitor &visitor) const override;
};

class NDouble : public NExpression {
public:
  double value;
  NDouble(double value) : value(value) {}
  void accept(Visitor &visitor) const override;
};

class NIdentifier : public NExpression {
public:
  std::string name;
  NIdentifier(const std::string &name) : name(name) {}
  void accept(Visitor &visitor) const override;
};

class NMethodCall : public NExpression {
public:
  const NIdentifier &id;
  ExpressionList arguments;
  NMethodCall(const NIdentifier &id, ExpressionList &arguments)
      : id(id), arguments(arguments) {}
  NMethodCall(const NIdentifier &id) : id(id) {}
  void accept(Visitor &visitor) const override;
};

class NBinaryOperator : public NExpression {
public:
  int op;
  NExpression &lhs;
  NExpression &rhs;
  NBinaryOperator(NExpression &lhs, int op, NExpression &rhs)
      : op(op), lhs(lhs), rhs(rhs) {}
  void accept(Visitor &visitor) const override;
};

class NAssignment : public NExpression {
public:
  NIdentifier &lhs;
  NExpression &rhs;
  NAssignment(NIdentifier &lhs, NExpression &rhs) : lhs(lhs), rhs(rhs) {}
  void accept(Visitor &visitor) const override;
};

class NBlock : public NExpression {
public:
  StatementList statements;
  NBlock() {}
  void accept(Visitor &visitor) const override;
};

class NIfExpression : public NExpression {
public:
  NExpression &condition;
  NExpression &thenExpr;
  NExpression &elseExpr;
  NIfExpression(NExpression &condition, NExpression &thenExpr,
                NExpression &elseExpr)
      : condition(condition), thenExpr(thenExpr), elseExpr(elseExpr) {}
  void accept(Visitor &visitor) const override;
};

class NLetExpression : public NExpression {
public:
  VariableList bindings;
  NExpression &body;
  NLetExpression(VariableList &bindings, NExpression &body)
      : bindings(bindings), body(body) {}
  void accept(Visitor &visitor) const override;
};

class NExpressionStatement : public NStatement {
public:
  NExpression &expression;
  NExpressionStatement(NExpression &expression) : expression(expression) {}
  void accept(Visitor &visitor) const override;
};

class NVariableDeclaration : public NStatement {
public:
  const NIdentifier &type;
  NIdentifier &id;
  NExpression *assignmentExpr;
  NVariableDeclaration(const NIdentifier &type, NIdentifier &id)
      : type(type), id(id), assignmentExpr(nullptr) {}
  NVariableDeclaration(const NIdentifier &type, NIdentifier &id,
                       NExpression *assignmentExpr)
      : type(type), id(id), assignmentExpr(assignmentExpr) {}
  void accept(Visitor &visitor) const override;
};

class NFunctionDeclaration : public NStatement {
public:
  const NIdentifier &type;
  const NIdentifier &id;
  VariableList arguments;
  NBlock &block;
  NFunctionDeclaration(const NIdentifier &type, const NIdentifier &id,
                       const VariableList &arguments, NBlock &block)
      : type(type), id(id), arguments(arguments), block(block) {}
  void accept(Visitor &visitor) const override;
};
// clang-format on

#endif // POLANG_NODE_HPP
