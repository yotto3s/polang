#ifndef POLANG_AST_PRINTER_HPP
#define POLANG_AST_PRINTER_HPP

#include <ostream>
#include <string>
#include <vector>

#include "parser/visitor.hpp"

class ASTPrinter : public Visitor {
public:
  explicit ASTPrinter(std::ostream& out) noexcept;

  void print(const NBlock& root);

  void visit(const NInteger& node) override;
  void visit(const NDouble& node) override;
  void visit(const NBoolean& node) override;
  void visit(const NIdentifier& node) override;
  void visit(const NMethodCall& node) override;
  void visit(const NBinaryOperator& node) override;
  void visit(const NAssignment& node) override;
  void visit(const NBlock& node) override;
  void visit(const NIfExpression& node) override;
  void visit(const NLetExpression& node) override;
  void visit(const NExpressionStatement& node) override;
  void visit(const NVariableDeclaration& node) override;
  void visit(const NFunctionDeclaration& node) override;

private:
  std::ostream& out_;
  std::vector<bool> depth_has_more_;

  void printPrefix() const;
  static std::string operatorToString(int op) noexcept;

  class DepthScope {
  public:
    DepthScope(ASTPrinter& printer, bool has_more) noexcept;
    ~DepthScope() noexcept;

  private:
    ASTPrinter& printer_;
  };
};

#endif // POLANG_AST_PRINTER_HPP
