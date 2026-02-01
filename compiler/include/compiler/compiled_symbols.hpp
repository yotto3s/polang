#ifndef POLANG_COMPILED_SYMBOLS_HPP
#define POLANG_COMPILED_SYMBOLS_HPP

#include "parser/type_inference.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

class NStatement;

namespace polang {

/// Tracks a compiled function's metadata for cross-module references.
struct CompiledFunction {
  std::string name;
  std::string returnType; // Type name string (e.g., "i64", "f64")
  std::vector<std::string> paramTypes;
  std::vector<std::string> paramNames;
  std::vector<std::string> captureNames;
  std::vector<std::string> captureTypes;
  bool isGeneric = false;
  std::vector<std::string> typeParams; // e.g. ["'a", "'b"]
  std::map<std::string, std::set<polang::TraitBound>> typeParamBounds;
};

/// Tracks a compiled global variable for cross-module references.
struct CompiledGlobal {
  std::string name;
  std::string type; // Type name string (e.g., "i64", "f64")
};

/// Registry of all previously compiled symbols, used to generate
/// extern declarations in incremental MLIR modules.
struct CompiledSymbols {
  std::map<std::string, CompiledFunction> functions;
  std::map<std::string, CompiledGlobal> globals;
  // AST nodes for generic functions (needed for re-emission since
  // monomorphization erases GenericFuncOps after specialization)
  std::vector<std::unique_ptr<NStatement>> genericFuncAstNodes;
  // Import mappings: local name → mangled name (e.g., "sub" → "Math$$sub")
  std::map<std::string, std::string> importMappings;
};

} // namespace polang

#endif // POLANG_COMPILED_SYMBOLS_HPP
