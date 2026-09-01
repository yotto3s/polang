//===- PolangTypes.td - Polang type definitions ------------*- tablegen -*-===//
//
// Defines the types for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TYPES_TD
#define POLANG_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "polang/Dialect/PolangDialect.td"

//===----------------------------------------------------------------------===//
// Polang type definitions
//===----------------------------------------------------------------------===//

class Polang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Polang_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Generic and compound types
//===----------------------------------------------------------------------===//

def Polang_TypeParamType : Polang_Type<"TypeParam", "type_param"> {
  let summary = "Named type parameter for generic functions";
  let description = [{
    Represents a named type parameter in a generic function template.
    Uses ML-style naming convention ('a, 'b, etc.).
    Example: !polang.type_param<"a">
  }];
  let parameters = (ins StringRefParameter<"">:$name);
  let assemblyFormat = "`<` $name `>`";
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def Polang_AnyNumericOrParam : AnyTypeOf<[
    AnyInteger, AnyFloat, Index, Polang_TypeParamType
], "numeric type or type parameter">;

def Polang_AnyTypeOrParam : AnyTypeOf<[
    AnyInteger, AnyFloat, Index, AnyTuple, Polang_TypeParamType
], "any type or type parameter">;

def Polang_I1OrParamType : AnyTypeOf<[
    I1, Polang_TypeParamType
], "i1 (bool) or type parameter">;

#endif // POLANG_TYPES_TD
