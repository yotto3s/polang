//===- PolangDialect.td - Polang dialect definition --------*- tablegen -*-===//
//
// Defines the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_DIALECT_TD
#define POLANG_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Polang dialect definition
//===----------------------------------------------------------------------===//

def Polang_Dialect : Dialect {
  let name = "polang";
  let summary = "A dialect for the Polang programming language";
  let description = [{
    The Polang dialect provides operations and types for representing
    Polang programs at a high level before lowering to standard MLIR dialects.
  }];
  let cppNamespace = "::polang";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base Polang operation definition
//===----------------------------------------------------------------------===//

class Polang_Op<string mnemonic, list<Trait> traits = []> :
    Op<Polang_Dialect, mnemonic, traits>;

#endif // POLANG_DIALECT_TD
