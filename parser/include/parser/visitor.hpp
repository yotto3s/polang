#ifndef POLANG_VISITOR_HPP
#define POLANG_VISITOR_HPP

// Forward declarations of all node types
class NInteger;
class NDouble;
class NBoolean;
class NIdentifier;
class NQualifiedName;
class NMethodCall;
class NBinaryOperator;
class NCastExpression;
class NAssignment;
class NBlock;
class NIfExpression;
class NLetExpression;
class NExpressionStatement;
class NVariableDeclaration;
class NFunctionDeclaration;
class NModuleDeclaration;
class NImportStatement;

/// Base class for implementing the Visitor design pattern on the Polang AST.
///
/// The Visitor pattern allows operations to be defined on AST nodes without
/// modifying the node classes themselves. This is used throughout the compiler
/// for:
/// - **Type checking** (TypeChecker): Validates type correctness
/// - **Free variable collection** (FreeVariableCollector): Identifies captures
/// - **AST printing** (ASTPrinter): Produces a text representation of the AST
/// - **Code generation** (MLIRGenVisitor): Generates MLIR from the AST
///
/// ## How to Implement a Visitor
///
/// 1. Create a class that inherits from Visitor
/// 2. Override all pure virtual visit() methods
/// 3. Use node.accept(*this) to dispatch to the correct visit method
///
/// ## Example
///
/// ```cpp
/// class MyVisitor : public Visitor {
/// public:
///   void visit(const NInteger& node) override {
///     std::cout << "Visiting integer: " << node.value << "\n";
///   }
///   // ... implement all other visit methods
/// };
///
/// // Usage:
/// MyVisitor visitor;
/// ast->accept(visitor);  // Calls appropriate visit() method
/// ```
///
/// ## Traversal
///
/// By default, this base class does not automatically traverse child nodes.
/// Each visitor implementation must explicitly call accept() on child nodes
/// when traversal is needed:
///
/// ```cpp
/// void visit(const NBinaryOperator& node) override {
///   node.lhs.accept(*this);  // Visit left operand
///   node.rhs.accept(*this);  // Visit right operand
/// }
/// ```
class Visitor {
public:
  virtual ~Visitor() noexcept = default;

  /// @name Expression Visitors
  /// @{
  virtual void visit(const NInteger& node) = 0;
  virtual void visit(const NDouble& node) = 0;
  virtual void visit(const NBoolean& node) = 0;
  virtual void visit(const NIdentifier& node) = 0;
  virtual void visit(const NQualifiedName& node) = 0;
  virtual void visit(const NMethodCall& node) = 0;
  virtual void visit(const NBinaryOperator& node) = 0;
  virtual void visit(const NCastExpression& node) = 0;
  virtual void visit(const NAssignment& node) = 0;
  virtual void visit(const NBlock& node) = 0;
  virtual void visit(const NIfExpression& node) = 0;
  virtual void visit(const NLetExpression& node) = 0;
  /// @}

  /// @name Statement Visitors
  /// @{
  virtual void visit(const NExpressionStatement& node) = 0;
  virtual void visit(const NVariableDeclaration& node) = 0;
  virtual void visit(const NFunctionDeclaration& node) = 0;
  virtual void visit(const NModuleDeclaration& node) = 0;
  virtual void visit(const NImportStatement& node) = 0;
  /// @}
};

#endif // POLANG_VISITOR_HPP
