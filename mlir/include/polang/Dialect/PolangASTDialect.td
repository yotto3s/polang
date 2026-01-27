//===- PolangASTDialect.td - Polang AST dialect definition --*- tablegen -*-===//
//
// Defines the Polang AST dialect for AST-level constructs.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_AST_DIALECT_TD
#define POLANG_AST_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Polang AST dialect definition
//===----------------------------------------------------------------------===//

def PolangAST_Dialect : Dialect {
  let name = "polang_ast";
  let summary = "Polang AST-level dialect with type inference support";
  let description = [{
    The Polang AST dialect represents source-level constructs before semantic
    analysis. It includes:
    - Unresolved variable references (var_ref, qualified_ref)
    - Module and import declarations
    - Let-in expressions with variable bindings
    - Operations that can have type variables (for type inference)

    This dialect is used for direct MLIR generation from the parser, before
    name resolution and type inference resolve all references and types.
  }];
  let cppNamespace = "::polang::ast";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base Polang AST operation definition
//===----------------------------------------------------------------------===//

class PolangAST_Op<string mnemonic, list<Trait> traits = []> :
    Op<PolangAST_Dialect, mnemonic, traits>;

#endif // POLANG_AST_DIALECT_TD
