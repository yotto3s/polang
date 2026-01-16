#ifndef POLANG_NODE_HPP
#define POLANG_NODE_HPP

#include <iostream>
#include <string>
#include <vector>

// Forward declare LLVM types to avoid header conflicts
namespace llvm {
class Value;
}

class CodeGenContext;
class NStatement;
class NExpression;
class NVariableDeclaration;

typedef std::vector<NStatement *> StatementList;
typedef std::vector<NExpression *> ExpressionList;
typedef std::vector<NVariableDeclaration *> VariableList;

// clang-format off
class Node {
public:
  virtual ~Node() {}
  virtual llvm::Value *codeGen(CodeGenContext &context) { return nullptr; }
};

class NExpression : public Node {};

class NStatement : public Node {};

class NInteger : public NExpression {
public:
  long long value;
  NInteger(long long value) : value(value) {}
  llvm::Value *codeGen(CodeGenContext &context) override;
};

class NDouble : public NExpression {
public:
  double value;
  NDouble(double value) : value(value) {}
  llvm::Value *codeGen(CodeGenContext &context) override;
};

class NIdentifier : public NExpression {
public:
  std::string name;
  NIdentifier(const std::string &name) : name(name) {}
  llvm::Value *codeGen(CodeGenContext &context) override;
};

class NMethodCall : public NExpression {
public:
  const NIdentifier &id;
  ExpressionList arguments;
  NMethodCall(const NIdentifier &id, ExpressionList &arguments)
      : id(id), arguments(arguments) {}
  NMethodCall(const NIdentifier &id) : id(id) {}
  llvm::Value *codeGen(CodeGenContext &context) override;
};

class NBinaryOperator : public NExpression {
public:
  int op;
  NExpression &lhs;
  NExpression &rhs;
  NBinaryOperator(NExpression &lhs, int op, NExpression &rhs)
      : op(op), lhs(lhs), rhs(rhs) {}
  llvm::Value *codeGen(CodeGenContext &context) override;
};

class NAssignment : public NExpression {
public:
  NIdentifier &lhs;
  NExpression &rhs;
  NAssignment(NIdentifier &lhs, NExpression &rhs) : lhs(lhs), rhs(rhs) {}
  llvm::Value *codeGen(CodeGenContext &context) override;
};

class NBlock : public NExpression {
public:
  StatementList statements;
  NBlock() {}
  llvm::Value *codeGen(CodeGenContext &context) override;
};

class NIfExpression : public NExpression {
public:
  NExpression &condition;
  NExpression &thenExpr;
  NExpression &elseExpr;
  NIfExpression(NExpression &condition, NExpression &thenExpr,
                NExpression &elseExpr)
      : condition(condition), thenExpr(thenExpr), elseExpr(elseExpr) {}
  llvm::Value *codeGen(CodeGenContext &context) override;
};

class NExpressionStatement : public NStatement {
public:
  NExpression &expression;
  NExpressionStatement(NExpression &expression) : expression(expression) {}
  llvm::Value *codeGen(CodeGenContext &context) override;
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
  llvm::Value *codeGen(CodeGenContext &context) override;
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
  llvm::Value *codeGen(CodeGenContext &context) override;
};
// clang-format on

#endif // POLANG_NODE_HPP
