#ifndef POLANG_VISITOR_HPP
#define POLANG_VISITOR_HPP

// Forward declarations of all node types
class NInteger;
class NDouble;
class NBoolean;
class NIdentifier;
class NMethodCall;
class NBinaryOperator;
class NAssignment;
class NBlock;
class NIfExpression;
class NLetExpression;
class NExpressionStatement;
class NVariableDeclaration;
class NFunctionDeclaration;

class Visitor {
public:
  virtual ~Visitor() noexcept = default;

  // Expression visitors
  virtual void visit(const NInteger& node) = 0;
  virtual void visit(const NDouble& node) = 0;
  virtual void visit(const NBoolean& node) = 0;
  virtual void visit(const NIdentifier& node) = 0;
  virtual void visit(const NMethodCall& node) = 0;
  virtual void visit(const NBinaryOperator& node) = 0;
  virtual void visit(const NAssignment& node) = 0;
  virtual void visit(const NBlock& node) = 0;
  virtual void visit(const NIfExpression& node) = 0;
  virtual void visit(const NLetExpression& node) = 0;

  // Statement visitors
  virtual void visit(const NExpressionStatement& node) = 0;
  virtual void visit(const NVariableDeclaration& node) = 0;
  virtual void visit(const NFunctionDeclaration& node) = 0;
};

#endif // POLANG_VISITOR_HPP
