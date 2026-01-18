#ifndef POLANG_TYPE_CHECKER_HPP
#define POLANG_TYPE_CHECKER_HPP

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <parser/visitor.hpp>

class Node;
class NBlock;
class NExpression;
struct NLetBinding;

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
  void visit(const NRefExpression& node) override;
  void visit(const NDerefExpression& node) override;
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

  // Helper methods for NLetExpression type checking
  void collectSiblingVarTypes(
      const std::vector<std::unique_ptr<NLetBinding>>& bindings,
      std::map<std::string, std::string>& siblingTypes,
      std::map<std::string, bool>& siblingMutability);

  void typeCheckLetBindings(
      const std::vector<std::unique_ptr<NLetBinding>>& bindings,
      const std::map<std::string, std::string>& siblingTypes,
      const std::map<std::string, bool>& siblingMutability,
      const std::map<std::string, std::string>& savedLocals,
      const std::map<std::string, bool>& savedMutability,
      std::vector<std::string>& bindingTypes,
      std::vector<bool>& bindingMutability,
      std::vector<std::vector<std::string>>& funcParams);

  void addLetBindingsToScope(
      const std::vector<std::unique_ptr<NLetBinding>>& bindings,
      const std::vector<std::string>& bindingTypes,
      const std::vector<bool>& bindingMutability,
      const std::vector<std::vector<std::string>>& funcParams);

  // Helper methods for NVariableDeclaration type checking
  void typeCheckVarDeclNoInit(NVariableDeclaration& node,
                              const std::string& varName);
  void typeCheckVarDeclInferType(NVariableDeclaration& node,
                                 const std::string& varName,
                                 const std::string& exprType);
  void typeCheckVarDeclWithAnnotation(NVariableDeclaration& node,
                                      const std::string& varName,
                                      const std::string& exprType);

  // Helper methods for NImportStatement type checking
  void handleModuleImport(const NImportStatement& node);
  void handleModuleAliasImport(const NImportStatement& node);
  void handleItemsImport(const NImportStatement& node);
  void handleWildcardImport(const NImportStatement& node);

  // Deferred type inference for generic types
  // Variables with unresolved generic types (name -> generic type)
  std::map<std::string, std::string> unresolvedGenerics;

  // Track AST nodes for updating types later (name -> node pointer)
  std::map<std::string, NVariableDeclaration*> varDeclNodes;

  // Helper methods for deferred type resolution
  void resolveGenericVariable(const std::string& varName,
                              const std::string& concreteType);
  void propagateTypeToSource(const NExpression* expr,
                             const std::string& targetType);
  void resolveRemainingGenerics();
};

#endif // POLANG_TYPE_CHECKER_HPP
