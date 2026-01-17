//===- PolangTypes.td - Polang type definitions ------------*- tablegen -*-===//
//
// Defines the types for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TYPES_TD
#define POLANG_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "polang/Dialect/PolangDialect.td"

//===----------------------------------------------------------------------===//
// Polang type definitions
//===----------------------------------------------------------------------===//

class Polang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Polang_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Signedness enum for integer types
//===----------------------------------------------------------------------===//

def Polang_Signed : I32EnumAttrCase<"Signed", 0, "signed">;
def Polang_Unsigned : I32EnumAttrCase<"Unsigned", 1, "unsigned">;

def Polang_Signedness : I32EnumAttr<"Signedness", "integer signedness", [
    Polang_Signed, Polang_Unsigned
]> {
  let cppNamespace = "::polang";
}

//===----------------------------------------------------------------------===//
// Parameterized numeric types
//===----------------------------------------------------------------------===//

def Polang_IntegerType : Polang_Type<"Integer", "integer"> {
  let summary = "Integer type with configurable width and signedness";
  let description = [{
    Parameterized integer type supporting 8, 16, 32, and 64-bit widths,
    both signed and unsigned.

    Examples:
    ```mlir
    !polang.integer<32, signed>   ; i32
    !polang.integer<8, unsigned>  ; u8
    !polang.integer<64, signed>   ; i64
    ```
  }];

  let parameters = (ins
    "unsigned":$width,
    "Signedness":$signedness
  );
  let assemblyFormat = "`<` $width `,` $signedness `>`";

  let extraClassDeclaration = [{
    bool isSigned() const { return getSignedness() == Signedness::Signed; }
    bool isUnsigned() const { return getSignedness() == Signedness::Unsigned; }
  }];
}

def Polang_FloatType : Polang_Type<"Float", "float"> {
  let summary = "Floating-point type with configurable width";
  let description = [{
    Parameterized floating-point type supporting 32 and 64-bit widths.

    Examples:
    ```mlir
    !polang.float<32>  ; f32
    !polang.float<64>  ; f64
    ```
  }];

  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
}

def Polang_BoolType : Polang_Type<"Bool", "bool"> {
  let summary = "1-bit boolean type";
  let description = [{
    The Polang boolean type represents a true/false value.
    It maps to `bool` in Polang source code.
  }];
}

def Polang_TypeVarType : Polang_Type<"TypeVar", "typevar"> {
  let summary = "Type variable for polymorphic types";
  let description = [{
    Represents an unresolved type variable used during type inference.
    The parameter is a unique identifier for the type variable.

    Example:
    ```mlir
    %x : !polang.typevar<0>
    ```
  }];

  let parameters = (ins "uint64_t":$id);
  let assemblyFormat = "`<` $id `>`";
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def Polang_AnyNumericType : AnyTypeOf<[Polang_IntegerType, Polang_FloatType],
    "Polang numeric type">;

def Polang_AnyType : AnyTypeOf<[Polang_IntegerType, Polang_FloatType, Polang_BoolType],
    "any Polang type">;

// Type constraints that include type variables (for use before type inference)
def Polang_AnyNumericOrVar : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_TypeVarType
], "Polang numeric type or type variable">;

def Polang_AnyTypeOrVar : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_BoolType, Polang_TypeVarType
], "any Polang type or type variable">;

// Bool or type variable (for if conditions before type inference)
def Polang_BoolOrTypeVar : AnyTypeOf<[
    Polang_BoolType, Polang_TypeVarType
], "bool or type variable">;

#endif // POLANG_TYPES_TD
