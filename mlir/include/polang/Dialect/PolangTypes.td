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
include "polang/Dialect/PolangEnums.td"

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

def Polang_IndexType : Polang_Type<"Index", "index", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
]> {
  let summary = "Pointer-width integer index type with signedness";
  let description = [{
    A pointer-width integer index type with explicit signedness semantics.

    Examples:
    ```mlir
    !polang.index<signed>    // isize
    !polang.index<unsigned>  // usize
    ```
  }];

  let parameters = (ins "polang::Signedness":$signedness);
  let assemblyFormat = "`<` $signedness `>`";

  let extraClassDeclaration = [{
    bool isSigned() const { return getSignedness() == polang::Signedness::Signed; }
    bool isUnsigned() const { return getSignedness() == polang::Signedness::Unsigned; }
  }];
}

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

def Polang_NonBoolInteger : Type<
    CPred<"::llvm::isa<::mlir::IntegerType>($_self) && ::llvm::cast<::mlir::IntegerType>($_self).getWidth() > 1">,
    "integer type with width > 1", "::mlir::IntegerType">;

def Polang_AnyNumericOrParam : AnyTypeOf<[
    Polang_NonBoolInteger, AnyFloat, Polang_IndexType, Index, Polang_TypeParamType
], "numeric type or type parameter">;

def Polang_AnyTypeOrParam : AnyTypeOf<[
    AnyInteger, AnyFloat, Polang_IndexType, Index, AnyTuple, Polang_TypeParamType
], "any type or type parameter">;

def Polang_I1OrParamType : AnyTypeOf<[
    I1, Polang_TypeParamType
], "i1 (bool) or type parameter">;

#endif // POLANG_TYPES_TD
