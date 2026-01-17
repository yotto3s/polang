#ifndef POLANG_NODE_HPP
#define POLANG_NODE_HPP

#include <string>
#include <utility>
#include <vector>

class Visitor;
class NStatement;
class NExpression;
class NVariableDeclaration;
class NFunctionDeclaration;

using StatementList = std::vector<NStatement*>;
using ExpressionList = std::vector<NExpression*>;
using VariableList = std::vector<NVariableDeclaration*>;
using StringList = std::vector<std::string>;

// Union type for let bindings (can be variable or function)
struct NLetBinding {
  bool isFunction;
  NVariableDeclaration* var;
  NFunctionDeclaration* func;
  NLetBinding(NVariableDeclaration* v)
      : isFunction(false), var(v), func(nullptr) {}
  NLetBinding(NFunctionDeclaration* f)
      : isFunction(true), var(nullptr), func(f) {}
};

using LetBindingList = std::vector<NLetBinding*>;

// clang-format off
class Node {
public:
  virtual ~Node() noexcept = default;
  virtual void accept(Visitor &visitor) const = 0;
};

class NExpression : public Node {};

class NStatement : public Node {};

class NInteger : public NExpression {
public:
  long long value;
  NInteger(long long value) : value(value) {}
  void accept(Visitor &visitor) const override;
};

class NDouble : public NExpression {
public:
  double value;
  NDouble(double value) : value(value) {}
  void accept(Visitor &visitor) const override;
};

class NBoolean : public NExpression {
public:
  bool value;
  NBoolean(bool value) : value(value) {}
  void accept(Visitor &visitor) const override;
};

class NIdentifier : public NExpression {
public:
  std::string name;
  NIdentifier(std::string name) : name(std::move(name)) {}
  void accept(Visitor &visitor) const override;
};

// Qualified name for module access: Math.add, Math.Internal.helper
class NQualifiedName : public NExpression {
public:
  StringList parts;  // ["Math", "add"] or ["Math", "Internal", "helper"]
  NQualifiedName(StringList parts) : parts(std::move(parts)) {}
  // Convenience constructor from single identifier
  NQualifiedName(std::string name) : parts({std::move(name)}) {}
  // Get the full qualified name as a string (e.g., "Math.add")
  [[nodiscard]] std::string fullName() const {
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) {
        result += ".";
      }
      result += parts[i];
    }
    return result;
  }
  // Get mangled name for MLIR (e.g., "Math$$add")
  [[nodiscard]] std::string mangledName() const {
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) {
        result += "$$";
      }
      result += parts[i];
    }
    return result;
  }
  // Check if this is a simple (unqualified) name
  [[nodiscard]] bool isSimple() const { return parts.size() == 1; }
  // Get the simple name (last part)
  [[nodiscard]] const std::string &simpleName() const { return parts.back(); }
  void accept(Visitor &visitor) const override;
};

class NMethodCall : public NExpression {
public:
  const NIdentifier &id;          // For backward compatibility
  const NQualifiedName *qualifiedId;  // For qualified calls (optional)
  ExpressionList arguments;
  NMethodCall(const NIdentifier &id, const ExpressionList &arguments)
      : id(id), qualifiedId(nullptr), arguments(arguments) {}
  NMethodCall(const NIdentifier &id) : id(id), qualifiedId(nullptr) {}
  // Constructor for qualified calls
  NMethodCall(const NQualifiedName &qid, const ExpressionList &arguments)
      : id(*new NIdentifier(qid.simpleName())), qualifiedId(&qid), arguments(arguments) {}
  // Get the effective function name (mangled if qualified)
  [[nodiscard]] std::string effectiveName() const {
    if (qualifiedId != nullptr) {
      return qualifiedId->mangledName();
    }
    return id.name;
  }
  void accept(Visitor &visitor) const override;
};

class NBinaryOperator : public NExpression {
public:
  int op;
  const NExpression &lhs;
  const NExpression &rhs;
  NBinaryOperator(const NExpression &lhs, int op, const NExpression &rhs)
      : op(op), lhs(lhs), rhs(rhs) {}
  void accept(Visitor &visitor) const override;
};

class NAssignment : public NExpression {
public:
  const NIdentifier &lhs;
  const NExpression &rhs;
  NAssignment(const NIdentifier &lhs, const NExpression &rhs) : lhs(lhs), rhs(rhs) {}
  void accept(Visitor &visitor) const override;
};

class NBlock : public NExpression {
public:
  StatementList statements;
  NBlock() = default;
  void accept(Visitor &visitor) const override;
};

class NIfExpression : public NExpression {
public:
  const NExpression &condition;
  const NExpression &thenExpr;
  const NExpression &elseExpr;
  NIfExpression(const NExpression &condition, const NExpression &thenExpr,
                const NExpression &elseExpr)
      : condition(condition), thenExpr(thenExpr), elseExpr(elseExpr) {}
  void accept(Visitor &visitor) const override;
};

class NLetExpression : public NExpression {
public:
  LetBindingList bindings;
  const NExpression &body;
  NLetExpression(const LetBindingList &bindings, const NExpression &body)
      : bindings(bindings), body(body) {}
  void accept(Visitor &visitor) const override;
};

class NExpressionStatement : public NStatement {
public:
  const NExpression &expression;
  NExpressionStatement(const NExpression &expression) : expression(expression) {}
  void accept(Visitor &visitor) const override;
};

class NVariableDeclaration : public NStatement {
public:
  NIdentifier *type;  // nullptr when type should be inferred
  NIdentifier &id;
  NExpression *assignmentExpr;
  bool isMutable;  // true for 'let mut', false for 'let'
  // Constructor for inferred type (no annotation)
  NVariableDeclaration(NIdentifier &id, NExpression *assignmentExpr,
                       bool isMutable = false)
      : type(nullptr), id(id), assignmentExpr(assignmentExpr),
        isMutable(isMutable) {}
  // Constructor for explicit type annotation
  NVariableDeclaration(NIdentifier *type, NIdentifier &id,
                       NExpression *assignmentExpr, bool isMutable = false)
      : type(type), id(id), assignmentExpr(assignmentExpr),
        isMutable(isMutable) {}
  void accept(Visitor &visitor) const override;
};

class NFunctionDeclaration : public NStatement {
public:
  NIdentifier *type;  // nullptr when return type should be inferred
  const NIdentifier &id;
  VariableList arguments;
  NBlock &block;
  VariableList captures;  // captured variables (filled by type checker)
  // Constructor for inferred return type
  NFunctionDeclaration(const NIdentifier &id, const VariableList &arguments,
                       NBlock &block)
      : type(nullptr), id(id), arguments(arguments), block(block) {}
  // Constructor for explicit return type
  NFunctionDeclaration(NIdentifier *type, const NIdentifier &id,
                       const VariableList &arguments, NBlock &block)
      : type(type), id(id), arguments(arguments), block(block) {}
  void accept(Visitor &visitor) const override;
};

class NModuleDeclaration : public NStatement {
public:
  const NIdentifier &name;
  StringList exports;        // Haskell-style export list from module header
  StatementList members;     // Functions, variables, nested modules
  // Constructor with exports
  NModuleDeclaration(const NIdentifier &name, StringList exports,
                     StatementList members)
      : name(name), exports(std::move(exports)), members(std::move(members)) {}
  // Constructor without exports (all private)
  NModuleDeclaration(const NIdentifier &name, StatementList members)
      : name(name), members(std::move(members)) {}
  void accept(Visitor &visitor) const override;
};

// Import kinds for different import statement forms
enum class ImportKind {
  Module,      // import Math (use as Math.add)
  ModuleAlias, // import Math as M (use as M.add)
  Items,       // from Math import add, PI
  All          // from Math import *
};

// Item in a "from X import a, b as c" statement
struct ImportItem {
  std::string name;   // Original name in the module
  std::string alias;  // Alias (empty if no alias)
  ImportItem(std::string name, std::string alias = "")
      : name(std::move(name)), alias(std::move(alias)) {}
  // Get the effective name to use in code
  [[nodiscard]] std::string effectiveName() const {
    return alias.empty() ? name : alias;
  }
};

using ImportItemList = std::vector<ImportItem>;

class NImportStatement : public NStatement {
public:
  ImportKind kind;
  const NQualifiedName &modulePath;  // Module being imported
  std::string alias;                  // For "import X as Y"
  ImportItemList items;               // For "from X import a, b"
  // Constructor for "import Math"
  NImportStatement(const NQualifiedName &modulePath)
      : kind(ImportKind::Module), modulePath(modulePath) {}
  // Constructor for "import Math as M"
  NImportStatement(const NQualifiedName &modulePath, std::string alias)
      : kind(ImportKind::ModuleAlias), modulePath(modulePath),
        alias(std::move(alias)) {}
  // Constructor for "from Math import add, PI" or "from Math import *"
  NImportStatement(const NQualifiedName &modulePath, ImportItemList items,
                   bool importAll = false)
      : kind(importAll ? ImportKind::All : ImportKind::Items),
        modulePath(modulePath), items(std::move(items)) {}
  void accept(Visitor &visitor) const override;
};
// clang-format on

#endif // POLANG_NODE_HPP
