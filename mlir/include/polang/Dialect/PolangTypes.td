//===- PolangTypes.td - Polang type definitions ------------*- tablegen -*-===//
//
// Defines the types for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TYPES_TD
#define POLANG_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "polang/Dialect/PolangDialect.td"

//===----------------------------------------------------------------------===//
// Polang type definitions
//===----------------------------------------------------------------------===//

class Polang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Polang_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Polang_IntType : Polang_Type<"Int", "int"> {
  let summary = "64-bit signed integer type";
  let description = [{
    The Polang integer type represents a 64-bit signed integer value.
    It maps to `int` in Polang source code.
  }];
}

def Polang_DoubleType : Polang_Type<"Double", "double"> {
  let summary = "64-bit floating-point type";
  let description = [{
    The Polang double type represents a 64-bit floating-point value.
    It maps to `double` in Polang source code.
  }];
}

def Polang_BoolType : Polang_Type<"Bool", "bool"> {
  let summary = "1-bit boolean type";
  let description = [{
    The Polang boolean type represents a true/false value.
    It maps to `bool` in Polang source code.
  }];
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def Polang_AnyNumericType : AnyTypeOf<[Polang_IntType, Polang_DoubleType],
    "Polang numeric type">;

def Polang_AnyType : AnyTypeOf<[Polang_IntType, Polang_DoubleType, Polang_BoolType],
    "any Polang type">;

#endif // POLANG_TYPES_TD
