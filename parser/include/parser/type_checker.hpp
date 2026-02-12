#ifndef POLANG_TYPE_CHECKER_HPP
#define POLANG_TYPE_CHECKER_HPP

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <parser/type_inference.hpp>
#include <parser/visitor.hpp>

class Node;
class NBlock;
class NExpression;
class NTypeSpec;
class NArrowType;
class NProductType;
class NFunctionDeclaration;
struct NLetBinding;
struct SourceLocation;

struct TypeCheckError {
  std::string message;
  int line;
  int column;
  TypeCheckError(std::string msg, int l = 0, int c = 0)
      : message(std::move(msg)), line(l), column(c) {}
};

// Snapshot of TypeChecker state for rollback on error
struct TypeCheckerSnapshot {
  std::map<std::string, std::string> localTypes;
  std::map<std::string, polang::FunctionSignature> functionSignatures;
  std::map<std::string, std::set<std::string>> moduleExports;
  std::map<std::string, std::string> moduleAliases;
  std::map<std::string, std::string> importedSymbols;
  std::map<std::string, std::unique_ptr<const NTypeSpec>> pendingTypeSignatures;
};

class TypeChecker : public Visitor {
public:
  TypeChecker();

  // Type Specification Visitor methods
  void visit(const NNamedType& node) override;
  void visit(const NArrowType& node) override;
  void visit(const NProductType& node) override;
  void visit(const NTypeVar& node) override;
  void visit(const NForallType& node) override;
  void visit(const NUnitType& node) override;

  // Expression Visitor methods
  void visit(const NInteger& node) override;
  void visit(const NDouble& node) override;
  void visit(const NBoolean& node) override;
  void visit(const NIdentifier& node) override;
  void visit(const NQualifiedName& node) override;
  void visit(const NMethodCall& node) override;
  void visit(const NBinaryOperator& node) override;
  void visit(const NCastExpression& node) override;
  void visit(const NBlock& node) override;
  void visit(const NIfExpression& node) override;
  void visit(const NLetExpression& node) override;
  void visit(const NUnitLiteral& node) override;
  void visit(const NExpressionStatement& node) override;
  void visit(const NVariableDeclaration& node) override;
  void visit(const NFunctionDeclaration& node) override;
  void visit(const NModuleDeclaration& node) override;
  void visit(const NImportStatement& node) override;
  void visit(const NTypeSignature& node) override;

  // Get the inferred type of the last visited node
  [[nodiscard]] std::string getInferredType() const noexcept {
    return inferredType;
  }

  // Retrieve all collected errors
  [[nodiscard]] const std::vector<TypeCheckError>& getErrors() const noexcept {
    return errors;
  }

  // Check if there were any errors
  [[nodiscard]] bool hasErrors() const noexcept { return !errors.empty(); }

  // Check an AST and return errors (resets all state)
  [[nodiscard]] std::vector<TypeCheckError> check(const NBlock& ast);

  // Incrementally check new statements, preserving persistent environment
  [[nodiscard]] std::vector<TypeCheckError>
  checkIncremental(const NBlock& newStatements);

  // Save current state for rollback on error
  [[nodiscard]] TypeCheckerSnapshot saveState() const;

  // Restore state from a snapshot
  void restoreState(const TypeCheckerSnapshot& snapshot);

private:
  std::string inferredType;
  std::map<std::string, std::string> localTypes;
  std::vector<TypeCheckError> errors;

  // Scope depth: 0 = top level, >0 = inside function body
  int scopeDepth = 0;

  // Module path for name mangling (e.g., ["Math", "Internal"])
  std::vector<std::string> modulePath;

  // Module exports: module mangled name -> set of exported symbol names
  std::map<std::string, std::set<std::string>> moduleExports;

  // Module aliases: alias -> original module path
  std::map<std::string, std::string> moduleAliases;

  // Imported symbols: local name -> mangled module symbol name
  std::map<std::string, std::string> importedSymbols;

  // Pending type signatures: name -> type expression
  std::map<std::string, std::unique_ptr<const NTypeSpec>> pendingTypeSignatures;

  // Apply a type signature to a function declaration
  void applyFunctionSignature(NFunctionDeclaration& node,
                              std::unique_ptr<const NTypeSpec> signature);

  // Parse and register a type signature for forward references
  // Returns true if the signature was successfully registered
  [[nodiscard]] bool registerTypeSignature(const std::string& name,
                                           const NTypeSpec& signature);

  // Warn about type signatures with no corresponding definition
  void warnOrphanedTypeSignatures();

  // Get mangled name for a symbol within current module context
  [[nodiscard]] std::string mangledName(const std::string& name) const;

  void reportError(const std::string& message);
  void reportError(const std::string& message, const SourceLocation& loc);

  // Collect identifiers referenced in a block that are not locally defined
  [[nodiscard]] std::set<std::string>
  collectFreeVariables(const NBlock& block,
                       const std::set<std::string>& localNames) const;

  // Helper methods for NLetExpression type checking
  void collectSiblingVarTypes(
      const std::vector<std::unique_ptr<NLetBinding>>& bindings,
      std::map<std::string, std::string>& siblingTypes);

  void typeCheckLetBindings(
      const std::vector<std::unique_ptr<NLetBinding>>& bindings,
      const std::map<std::string, std::string>& siblingTypes,
      const std::map<std::string, std::string>& savedLocals,
      std::vector<std::string>& bindingTypes,
      std::vector<std::vector<std::string>>& funcParams);

  void addLetBindingsToScope(
      const std::vector<std::unique_ptr<NLetBinding>>& bindings,
      const std::vector<std::string>& bindingTypes,
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

  // Helper methods for NBinaryOperator type checking
  void checkArithmeticBinaryOp(const NBinaryOperator& node,
                               const std::string& lhsType,
                               const std::string& rhsType);
  void checkComparisonBinaryOp(const NBinaryOperator& node,
                               const std::string& lhsType,
                               const std::string& rhsType);

  // Track AST nodes for variables with generic types needing deferred
  // resolution
  std::map<std::string, NVariableDeclaration*> varDeclNodes;

  // Helper methods for deferred type resolution
  void resolveGenericVariable(const std::string& varName,
                              const std::string& concreteType);
  void propagateTypeToSource(const NExpression* expr,
                             const std::string& targetType);
  void resolveRemainingGenerics();

  // HM type inference infrastructure
  polang::Substitution subst;
  polang::Unifier unifier;
  polang::TraitConstraints traitConstraints;
  // Map function name -> signature (monomorphic or polymorphic)
  std::map<std::string, polang::FunctionSignature> functionSignatures;

  // Infer a function: assign unification vars, collect constraints, resolve
  void inferFunction(NFunctionDeclaration& node, const std::string& funcName,
                     const std::map<std::string, std::string>& savedLocals);

  // Instantiate a polymorphic function at a call site
  void instantiateCall(NMethodCall& node, const std::string& funcName,
                       const polang::PolymorphicSignature& scheme,
                       const std::vector<std::string>& argTypes);

  // Validate type names in a type expression tree.
  // Returns false if any errors were found.
  // If usedTypeVars is provided, collects all NTypeVar references found.
  bool
  validateTypeNames(const NTypeSpec& typeSpec,
                    const std::set<std::string>& declaredTypeVars,
                    std::optional<std::reference_wrapper<std::set<std::string>>>
                        usedTypeVars = std::nullopt);
};

#endif // POLANG_TYPE_CHECKER_HPP
