#ifndef POLANG_VISITOR_HPP
#define POLANG_VISITOR_HPP

// Forward declarations of all node types
class NInteger;
class NDouble;
class NIdentifier;
class NMethodCall;
class NBinaryOperator;
class NAssignment;
class NBlock;
class NIfExpression;
class NExpressionStatement;
class NVariableDeclaration;
class NFunctionDeclaration;

class Visitor {
public:
  virtual ~Visitor() = default;

  // Expression visitors
  virtual void visit(NInteger& node) = 0;
  virtual void visit(NDouble& node) = 0;
  virtual void visit(NIdentifier& node) = 0;
  virtual void visit(NMethodCall& node) = 0;
  virtual void visit(NBinaryOperator& node) = 0;
  virtual void visit(NAssignment& node) = 0;
  virtual void visit(NBlock& node) = 0;
  virtual void visit(NIfExpression& node) = 0;

  // Statement visitors
  virtual void visit(NExpressionStatement& node) = 0;
  virtual void visit(NVariableDeclaration& node) = 0;
  virtual void visit(NFunctionDeclaration& node) = 0;
};

#endif // POLANG_VISITOR_HPP
