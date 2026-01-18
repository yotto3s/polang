#ifndef POLANG_TYPE_CHECKER_HPP
#define POLANG_TYPE_CHECKER_HPP

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <parser/visitor.hpp>

class Node;
class NBlock;

struct TypeCheckError {
  std::string message;
  int line;
  int column;
  TypeCheckError(std::string msg, int l = 0, int c = 0)
      : message(std::move(msg)), line(l), column(c) {}
};

class TypeChecker : public Visitor {
public:
  TypeChecker();

  // Visitor methods
  void visit(const NInteger& node) override;
  void visit(const NDouble& node) override;
  void visit(const NBoolean& node) override;
  void visit(const NIdentifier& node) override;
  void visit(const NQualifiedName& node) override;
  void visit(const NMethodCall& node) override;
  void visit(const NBinaryOperator& node) override;
  void visit(const NCastExpression& node) override;
  void visit(const NAssignment& node) override;
  void visit(const NBlock& node) override;
  void visit(const NIfExpression& node) override;
  void visit(const NLetExpression& node) override;
  void visit(const NExpressionStatement& node) override;
  void visit(const NVariableDeclaration& node) override;
  void visit(const NFunctionDeclaration& node) override;
  void visit(const NModuleDeclaration& node) override;
  void visit(const NImportStatement& node) override;

  // Get the inferred type of the last visited node
  [[nodiscard]] std::string getInferredType() const { return inferredType; }

  // Retrieve all collected errors
  [[nodiscard]] const std::vector<TypeCheckError>& getErrors() const {
    return errors;
  }

  // Check if there were any errors
  [[nodiscard]] bool hasErrors() const { return !errors.empty(); }

  // Check an AST and return errors
  std::vector<TypeCheckError> check(const NBlock& ast);

private:
  std::string inferredType;
  std::map<std::string, std::string> localTypes;
  std::map<std::string, bool> localMutability;
  std::map<std::string, std::string> functionReturnTypes;
  std::map<std::string, std::vector<std::string>> functionParamTypes;
  std::vector<TypeCheckError> errors;

  // Module path for name mangling (e.g., ["Math", "Internal"])
  std::vector<std::string> modulePath;

  // Module exports: module mangled name -> set of exported symbol names
  std::map<std::string, std::set<std::string>> moduleExports;

  // Module aliases: alias -> original module path
  std::map<std::string, std::string> moduleAliases;

  // Imported symbols: local name -> mangled module symbol name
  std::map<std::string, std::string> importedSymbols;

  // Get mangled name for a symbol within current module context
  [[nodiscard]] std::string mangledName(const std::string& name) const;

  void reportError(const std::string& message);

  // Collect identifiers referenced in a block that are not locally defined
  [[nodiscard]] std::set<std::string>
  collectFreeVariables(const NBlock& block,
                       const std::set<std::string>& localNames) const;
};

#endif // POLANG_TYPE_CHECKER_HPP
