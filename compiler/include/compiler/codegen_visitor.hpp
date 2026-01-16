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
  void visit(NInteger& node) override;
  void visit(NDouble& node) override;
  void visit(NIdentifier& node) override;
  void visit(NMethodCall& node) override;
  void visit(NBinaryOperator& node) override;
  void visit(NAssignment& node) override;
  void visit(NBlock& node) override;
  void visit(NIfExpression& node) override;

  // Statement visitors
  void visit(NExpressionStatement& node) override;
  void visit(NVariableDeclaration& node) override;
  void visit(NFunctionDeclaration& node) override;

private:
  CodeGenContext& context_;
  llvm::Value* result_ = nullptr;

  // Helper to visit a node and return its result
  llvm::Value* generate(Node& node);
};

#endif // POLANG_CODEGEN_VISITOR_HPP
