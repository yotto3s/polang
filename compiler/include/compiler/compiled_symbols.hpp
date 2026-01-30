#ifndef POLANG_COMPILED_SYMBOLS_HPP
#define POLANG_COMPILED_SYMBOLS_HPP

#include <map>
#include <string>

namespace polang {

/// Tracks a compiled function's metadata for cross-module references.
struct CompiledFunction {
  std::string name;
  std::string returnType; // Type name string (e.g., "i64", "f64")
  std::vector<std::string> paramTypes;
  bool isGeneric = false;
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
};

} // namespace polang

#endif // POLANG_COMPILED_SYMBOLS_HPP
