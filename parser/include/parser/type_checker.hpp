#ifndef POLANG_TYPE_CHECKER_HPP
#define POLANG_TYPE_CHECKER_HPP

#include <map>
#include <set>
#include <string>
#include <vector>

#include <parser/visitor.hpp>

class Node;
class NBlock;

struct TypeCheckError {
  std::string message;
  int line;
  int column;
  TypeCheckError(const std::string& msg, int l = 0, int c = 0)
      : message(msg), line(l), column(c) {}
};

class TypeChecker : public Visitor {
public:
  TypeChecker();

  // Visitor methods
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

  // Get the inferred type of the last visited node
  std::string getInferredType() const { return inferred_type_; }

  // Retrieve all collected errors
  const std::vector<TypeCheckError>& getErrors() const { return errors_; }

  // Check if there were any errors
  bool hasErrors() const { return !errors_.empty(); }

  // Check an AST and return errors
  std::vector<TypeCheckError> check(const NBlock& ast);

private:
  std::string inferred_type_;
  std::map<std::string, std::string> local_types_;
  std::map<std::string, bool> local_mutability_;
  std::map<std::string, std::string> function_return_types_;
  std::map<std::string, std::vector<std::string>> function_param_types_;
  std::vector<TypeCheckError> errors_;

  void reportError(const std::string& message);
  std::string inferType(const Node& node);

  // Collect identifiers referenced in a block that are not locally defined
  std::set<std::string>
  collectFreeVariables(const NBlock& block,
                       const std::set<std::string>& local_names) const;
};

#endif // POLANG_TYPE_CHECKER_HPP
