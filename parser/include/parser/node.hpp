#ifndef POLANG_NODE_HPP
#define POLANG_NODE_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

class Visitor;
class NStatement;
class NExpression;
class NVariableDeclaration;
class NFunctionDeclaration;
class NIdentifier;

// Source location information for error reporting
struct SourceLocation {
  int line = 0;
  int column = 0;
  SourceLocation() = default;
  SourceLocation(int l, int c) : line(l), column(c) {}
  [[nodiscard]] bool isValid() const { return line > 0; }
};

// Smart pointer type aliases for owning containers
using StatementList = std::vector<std::unique_ptr<NStatement>>;
using ExpressionList = std::vector<std::unique_ptr<NExpression>>;
using VariableList = std::vector<std::unique_ptr<NVariableDeclaration>>;
using StringList = std::vector<std::string>;

// Union type for let bindings (can be variable or function)
struct NLetBinding {
  bool isFunction;
  std::unique_ptr<NVariableDeclaration> var;
  std::unique_ptr<NFunctionDeclaration> func;
  explicit NLetBinding(std::unique_ptr<NVariableDeclaration> v)
      : isFunction(false), var(std::move(v)), func(nullptr) {}
  explicit NLetBinding(std::unique_ptr<NFunctionDeclaration> f)
      : isFunction(true), var(nullptr), func(std::move(f)) {}
};

using LetBindingList = std::vector<std::unique_ptr<NLetBinding>>;

// clang-format off
class Node {
public:
  SourceLocation loc;
  virtual ~Node() noexcept = default;
  virtual void accept(Visitor &visitor) const = 0;
  void setLocation(int line, int column) { loc = SourceLocation(line, column); }
};

class NExpression : public Node {};

class NStatement : public Node {};

class NInteger : public NExpression {
public:
  long long value;
  explicit NInteger(long long value) : value(value) {}
  void accept(Visitor &visitor) const override;
};

class NDouble : public NExpression {
public:
  double value;
  explicit NDouble(double value) : value(value) {}
  void accept(Visitor &visitor) const override;
};

class NBoolean : public NExpression {
public:
  bool value;
  explicit NBoolean(bool value) : value(value) {}
  void accept(Visitor &visitor) const override;
};

class NIdentifier : public NExpression {
public:
  std::string name;
  explicit NIdentifier(std::string name) : name(std::move(name)) {}
  void accept(Visitor &visitor) const override;
};

// Forward declaration
class NTypeSpec;

// Base class for type specifications
class NTypeSpec : public Node {
public:
  // Get the string representation of this type (for backwards compatibility)
  [[nodiscard]] virtual std::string getTypeName() const = 0;
};

// Named type (base types like i64, f64, bool, typevar)
class NNamedType : public NTypeSpec {
public:
  std::string name;
  explicit NNamedType(std::string name) : name(std::move(name)) {}
  [[nodiscard]] std::string getTypeName() const override { return name; }
  void accept(Visitor &visitor) const override;
};

// Reference type: ref T
class NRefType : public NTypeSpec {
public:
  std::shared_ptr<const NTypeSpec> innerType;
  explicit NRefType(std::shared_ptr<const NTypeSpec> inner) : innerType(std::move(inner)) {}
  [[nodiscard]] std::string getTypeName() const override {
    return "ref " + innerType->getTypeName();
  }
  void accept(Visitor &visitor) const override;
};

// Mutable reference type: mut T
class NMutRefType : public NTypeSpec {
public:
  std::shared_ptr<const NTypeSpec> innerType;
  explicit NMutRefType(std::shared_ptr<const NTypeSpec> inner) : innerType(std::move(inner)) {}
  [[nodiscard]] std::string getTypeName() const override {
    return "mut " + innerType->getTypeName();
  }
  void accept(Visitor &visitor) const override;
};

// Capture entry for closures (owns its type and id via shared_ptr/unique_ptr)
// Mutability is derived from the type annotation (e.g., "mut i64" prefix)
struct CaptureEntry {
  std::shared_ptr<const NTypeSpec> type;
  std::unique_ptr<NIdentifier> id;
  CaptureEntry(std::shared_ptr<const NTypeSpec> type, std::unique_ptr<NIdentifier> id)
      : type(std::move(type)), id(std::move(id)) {}
};

// Qualified name for module access: Math.add, Math.Internal.helper
class NQualifiedName : public NExpression {
public:
  StringList parts;  // ["Math", "add"] or ["Math", "Internal", "helper"]
  explicit NQualifiedName(StringList parts) : parts(std::move(parts)) {}
  // Convenience constructor from single identifier
  explicit NQualifiedName(std::string name) : parts({std::move(name)}) {}
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
  std::unique_ptr<NIdentifier> id;          // For simple calls
  std::unique_ptr<NQualifiedName> qualifiedId;  // For qualified calls (optional)
  ExpressionList arguments;
  NMethodCall(std::unique_ptr<NIdentifier> id, ExpressionList arguments)
      : id(std::move(id)), qualifiedId(nullptr), arguments(std::move(arguments)) {}
  explicit NMethodCall(std::unique_ptr<NIdentifier> id)
      : id(std::move(id)), qualifiedId(nullptr) {}
  // Constructor for qualified calls
  NMethodCall(std::unique_ptr<NQualifiedName> qid, ExpressionList arguments)
      : id(std::make_unique<NIdentifier>(qid->simpleName())),
        qualifiedId(std::move(qid)), arguments(std::move(arguments)) {}
  // Get the effective function name (mangled if qualified)
  [[nodiscard]] std::string effectiveName() const {
    if (qualifiedId != nullptr) {
      return qualifiedId->mangledName();
    }
    return id->name;
  }
  void accept(Visitor &visitor) const override;
};

class NBinaryOperator : public NExpression {
public:
  int op;
  std::unique_ptr<NExpression> lhs;
  std::unique_ptr<NExpression> rhs;
  NBinaryOperator(std::unique_ptr<NExpression> lhs, int op,
                  std::unique_ptr<NExpression> rhs)
      : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  void accept(Visitor &visitor) const override;
};

class NCastExpression : public NExpression {
public:
  std::unique_ptr<NExpression> expression;
  std::shared_ptr<const NTypeSpec> targetType;
  NCastExpression(std::unique_ptr<NExpression> expr,
                  std::shared_ptr<const NTypeSpec> type)
      : expression(std::move(expr)), targetType(std::move(type)) {}
  void accept(Visitor &visitor) const override;
};

class NAssignment : public NExpression {
public:
  std::unique_ptr<NIdentifier> lhs;
  std::unique_ptr<NExpression> rhs;
  NAssignment(std::unique_ptr<NIdentifier> lhs, std::unique_ptr<NExpression> rhs)
      : lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  void accept(Visitor &visitor) const override;
};

class NRefExpression : public NExpression {
public:
  std::unique_ptr<NExpression> expr;
  explicit NRefExpression(std::unique_ptr<NExpression> expr)
      : expr(std::move(expr)) {}
  void accept(Visitor &visitor) const override;
};

class NDerefExpression : public NExpression {
public:
  std::unique_ptr<NExpression> ref;
  explicit NDerefExpression(std::unique_ptr<NExpression> ref)
      : ref(std::move(ref)) {}
  void accept(Visitor &visitor) const override;
};

class NMutRefExpression : public NExpression {
public:
  std::unique_ptr<NExpression> expr;
  explicit NMutRefExpression(std::unique_ptr<NExpression> expr)
      : expr(std::move(expr)) {}
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
  std::unique_ptr<NExpression> condition;
  std::unique_ptr<NExpression> thenExpr;
  std::unique_ptr<NExpression> elseExpr;
  NIfExpression(std::unique_ptr<NExpression> condition,
                std::unique_ptr<NExpression> thenExpr,
                std::unique_ptr<NExpression> elseExpr)
      : condition(std::move(condition)), thenExpr(std::move(thenExpr)),
        elseExpr(std::move(elseExpr)) {}
  void accept(Visitor &visitor) const override;
};

class NLetExpression : public NExpression {
public:
  LetBindingList bindings;
  std::unique_ptr<NExpression> body;
  NLetExpression(LetBindingList bindings, std::unique_ptr<NExpression> body)
      : bindings(std::move(bindings)), body(std::move(body)) {}
  void accept(Visitor &visitor) const override;
};

class NExpressionStatement : public NStatement {
public:
  std::unique_ptr<NExpression> expression;
  explicit NExpressionStatement(std::unique_ptr<NExpression> expression)
      : expression(std::move(expression)) {}
  void accept(Visitor &visitor) const override;
};

class NVariableDeclaration : public NStatement {
public:
  std::shared_ptr<const NTypeSpec> type;  // nullptr when type should be inferred
  std::unique_ptr<NIdentifier> id;
  std::unique_ptr<NExpression> assignmentExpr;
  // Mutability is derived from type annotation (e.g., "mut i64" prefix)
  // or from NMutRefExpression in assignmentExpr
  // Constructor for inferred type (no annotation)
  NVariableDeclaration(std::unique_ptr<NIdentifier> id,
                       std::unique_ptr<NExpression> assignmentExpr)
      : type(nullptr), id(std::move(id)), assignmentExpr(std::move(assignmentExpr)) {}
  // Constructor for explicit type annotation
  NVariableDeclaration(std::shared_ptr<const NTypeSpec> type,
                       std::unique_ptr<NIdentifier> id,
                       std::unique_ptr<NExpression> assignmentExpr)
      : type(std::move(type)), id(std::move(id)),
        assignmentExpr(std::move(assignmentExpr)) {}
  void accept(Visitor &visitor) const override;
};

class NFunctionDeclaration : public NStatement {
public:
  std::shared_ptr<const NTypeSpec> type;  // nullptr when return type should be inferred
  std::unique_ptr<NIdentifier> id;
  VariableList arguments;
  std::unique_ptr<NBlock> block;
  std::vector<CaptureEntry> captures;  // captured variables (filled by type checker)
  // Constructor for inferred return type
  NFunctionDeclaration(std::unique_ptr<NIdentifier> id, VariableList arguments,
                       std::unique_ptr<NBlock> block)
      : type(nullptr), id(std::move(id)), arguments(std::move(arguments)),
        block(std::move(block)) {}
  // Constructor for explicit return type
  NFunctionDeclaration(std::shared_ptr<const NTypeSpec> type,
                       std::unique_ptr<NIdentifier> id, VariableList arguments,
                       std::unique_ptr<NBlock> block)
      : type(std::move(type)), id(std::move(id)), arguments(std::move(arguments)),
        block(std::move(block)) {}
  void accept(Visitor &visitor) const override;
};

class NModuleDeclaration : public NStatement {
public:
  std::unique_ptr<NIdentifier> name;
  StringList exports;        // Haskell-style export list from module header
  StatementList members;     // Functions, variables, nested modules
  // Constructor with exports
  NModuleDeclaration(std::unique_ptr<NIdentifier> name, StringList exports,
                     StatementList members)
      : name(std::move(name)), exports(std::move(exports)),
        members(std::move(members)) {}
  // Constructor without exports (all private)
  NModuleDeclaration(std::unique_ptr<NIdentifier> name, StatementList members)
      : name(std::move(name)), members(std::move(members)) {}
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
  std::unique_ptr<NQualifiedName> modulePath;  // Module being imported
  std::string alias;                  // For "import X as Y"
  ImportItemList items;               // For "from X import a, b"
  // Constructor for "import Math"
  explicit NImportStatement(std::unique_ptr<NQualifiedName> modulePath)
      : kind(ImportKind::Module), modulePath(std::move(modulePath)) {}
  // Constructor for "import Math as M"
  NImportStatement(std::unique_ptr<NQualifiedName> modulePath, std::string alias)
      : kind(ImportKind::ModuleAlias), modulePath(std::move(modulePath)),
        alias(std::move(alias)) {}
  // Constructor for "from Math import add, PI" or "from Math import *"
  NImportStatement(std::unique_ptr<NQualifiedName> modulePath, ImportItemList items,
                   bool importAll = false)
      : kind(importAll ? ImportKind::All : ImportKind::Items),
        modulePath(std::move(modulePath)), items(std::move(items)) {}
  void accept(Visitor &visitor) const override;
};
// clang-format on

#endif // POLANG_NODE_HPP
