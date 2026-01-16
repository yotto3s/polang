#ifndef POLANG_NODE_HPP
#define POLANG_NODE_HPP

#include <string>
#include <vector>

class Visitor;
class NStatement;
class NExpression;
class NVariableDeclaration;
class NFunctionDeclaration;

typedef std::vector<NStatement*> StatementList;
typedef std::vector<NExpression*> ExpressionList;
typedef std::vector<NVariableDeclaration*> VariableList;

// Union type for let bindings (can be variable or function)
struct NLetBinding {
  bool isFunction;
  NVariableDeclaration* var;
  NFunctionDeclaration* func;
  NLetBinding(NVariableDeclaration* v)
      : isFunction(false), var(v), func(nullptr) {}
  NLetBinding(NFunctionDeclaration* f)
      : isFunction(true), var(nullptr), func(f) {}
};

typedef std::vector<NLetBinding*> LetBindingList;

// clang-format off
class Node {
public:
  virtual ~Node() noexcept {}
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

class NBoolean : public NExpression {
public:
  bool value;
  NBoolean(bool value) : value(value) {}
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
  NMethodCall(const NIdentifier &id, const ExpressionList &arguments)
      : id(id), arguments(arguments) {}
  NMethodCall(const NIdentifier &id) : id(id) {}
  void accept(Visitor &visitor) const override;
};

class NBinaryOperator : public NExpression {
public:
  int op;
  const NExpression &lhs;
  const NExpression &rhs;
  NBinaryOperator(const NExpression &lhs, int op, const NExpression &rhs)
      : op(op), lhs(lhs), rhs(rhs) {}
  void accept(Visitor &visitor) const override;
};

class NAssignment : public NExpression {
public:
  const NIdentifier &lhs;
  const NExpression &rhs;
  NAssignment(const NIdentifier &lhs, const NExpression &rhs) : lhs(lhs), rhs(rhs) {}
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
  const NExpression &condition;
  const NExpression &thenExpr;
  const NExpression &elseExpr;
  NIfExpression(const NExpression &condition, const NExpression &thenExpr,
                const NExpression &elseExpr)
      : condition(condition), thenExpr(thenExpr), elseExpr(elseExpr) {}
  void accept(Visitor &visitor) const override;
};

class NLetExpression : public NExpression {
public:
  LetBindingList bindings;
  const NExpression &body;
  NLetExpression(const LetBindingList &bindings, const NExpression &body)
      : bindings(bindings), body(body) {}
  void accept(Visitor &visitor) const override;
};

class NExpressionStatement : public NStatement {
public:
  const NExpression &expression;
  NExpressionStatement(const NExpression &expression) : expression(expression) {}
  void accept(Visitor &visitor) const override;
};

class NVariableDeclaration : public NStatement {
public:
  NIdentifier *type;  // nullptr when type should be inferred
  NIdentifier &id;
  NExpression *assignmentExpr;
  // Constructor for inferred type (no annotation)
  NVariableDeclaration(NIdentifier &id, NExpression *assignmentExpr)
      : type(nullptr), id(id), assignmentExpr(assignmentExpr) {}
  // Constructor for explicit type annotation
  NVariableDeclaration(NIdentifier *type, NIdentifier &id,
                       NExpression *assignmentExpr)
      : type(type), id(id), assignmentExpr(assignmentExpr) {}
  void accept(Visitor &visitor) const override;
};

class NFunctionDeclaration : public NStatement {
public:
  NIdentifier *type;  // nullptr when return type should be inferred
  const NIdentifier &id;
  VariableList arguments;
  NBlock &block;
  // Constructor for inferred return type
  NFunctionDeclaration(const NIdentifier &id, const VariableList &arguments,
                       NBlock &block)
      : type(nullptr), id(id), arguments(arguments), block(block) {}
  // Constructor for explicit return type
  NFunctionDeclaration(NIdentifier *type, const NIdentifier &id,
                       const VariableList &arguments, NBlock &block)
      : type(type), id(id), arguments(arguments), block(block) {}
  void accept(Visitor &visitor) const override;
};
// clang-format on

#endif // POLANG_NODE_HPP
