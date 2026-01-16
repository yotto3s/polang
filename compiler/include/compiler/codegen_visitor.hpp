#ifndef POLANG_CODEGEN_VISITOR_HPP
#define POLANG_CODEGEN_VISITOR_HPP

#include <llvm/IR/Value.h>
#include <parser/visitor.hpp>

class CodeGenContext;
class Node;

class CodeGenVisitor : public Visitor {
public:
  explicit CodeGenVisitor(CodeGenContext& context);

  // Get result after visiting a node
  llvm::Value* getResult() const { return result_; }

  // Expression visitors
  void visit(const NInteger& node) override;
  void visit(const NDouble& node) override;
  void visit(const NIdentifier& node) override;
  void visit(const NMethodCall& node) override;
  void visit(const NBinaryOperator& node) override;
  void visit(const NAssignment& node) override;
  void visit(const NBlock& node) override;
  void visit(const NIfExpression& node) override;
  void visit(const NLetExpression& node) override;

  // Statement visitors
  void visit(const NExpressionStatement& node) override;
  void visit(const NVariableDeclaration& node) override;
  void visit(const NFunctionDeclaration& node) override;

private:
  CodeGenContext& context_;
  llvm::Value* result_ = nullptr;

  // Helper to visit a node and return its result
  llvm::Value* generate(const Node& node);
};

#endif // POLANG_CODEGEN_VISITOR_HPP
