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
// TypeVarKind enum for type variable constraints
//===----------------------------------------------------------------------===//

def Polang_TypeVarKind_Any : I32EnumAttrCase<"Any", 0, "any">;
def Polang_TypeVarKind_Integer : I32EnumAttrCase<"Integer", 1, "integer">;
def Polang_TypeVarKind_Float : I32EnumAttrCase<"Float", 2, "float">;

def Polang_TypeVarKind : I32EnumAttr<"TypeVarKind", "type variable kind", [
    Polang_TypeVarKind_Any, Polang_TypeVarKind_Integer, Polang_TypeVarKind_Float
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
  let summary = "Type variable for polymorphic type inference";
  let description = [{
    Represents an unresolved type that will be determined by type inference.
    The optional `kind` parameter constrains what concrete types this variable
    can unify with:
    - `any` (default): Can unify with any type (for polymorphic functions)
    - `integer`: Can only unify with integer types (for integer literals)
    - `float`: Can only unify with float types (for float literals)

    Examples:
    ```mlir
    !polang.typevar<1>          ; unconstrained type variable
    !polang.typevar<2, integer> ; must resolve to an integer type
    !polang.typevar<3, float>   ; must resolve to a float type
    ```
  }];

  let parameters = (ins
    "uint64_t":$id,
    DefaultValuedParameter<"TypeVarKind", "TypeVarKind::Any">:$kind
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool isAny() const { return getKind() == TypeVarKind::Any; }
    bool isIntegerKind() const { return getKind() == TypeVarKind::Integer; }
    bool isFloatKind() const { return getKind() == TypeVarKind::Float; }
  }];
}

//===----------------------------------------------------------------------===//
// Reference Types
//===----------------------------------------------------------------------===//

def Polang_MutRefType : Polang_Type<"MutRef", "mutref"> {
  let summary = "Mutable reference type";
  let description = [{
    A mutable reference type that allows both reading and writing.
    Created when declaring mutable variables with 'let x = mut value'.
    The element type is the type of the referenced value.

    Examples:
    ```mlir
    !polang.mutref<!polang.integer<64, signed>>  ; mut i64
    !polang.mutref<!polang.float<64>>            ; mut f64
    ```
  }];
  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def Polang_RefType : Polang_Type<"Ref", "ref"> {
  let summary = "Immutable reference type";
  let description = [{
    An immutable reference type that allows only reading.
    Created with 'ref expr' expressions.
    The element type is the type of the referenced value.

    Examples:
    ```mlir
    !polang.ref<!polang.integer<64, signed>>  ; ref i64
    !polang.ref<!polang.float<64>>            ; ref f64
    ```
  }];
  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
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
    Polang_IntegerType, Polang_FloatType, Polang_BoolType, Polang_TypeVarType,
    Polang_MutRefType, Polang_RefType
], "any Polang type or type variable">;

// Bool or type variable (for if conditions before type inference)
def Polang_BoolOrTypeVar : AnyTypeOf<[
    Polang_BoolType, Polang_TypeVarType
], "bool or type variable">;

// Integer or type variable (for integer constants before type inference)
def Polang_IntegerOrTypeVar : AnyTypeOf<[
    Polang_IntegerType, Polang_TypeVarType
], "integer or type variable">;

// Float or type variable (for float constants before type inference)
def Polang_FloatOrTypeVar : AnyTypeOf<[
    Polang_FloatType, Polang_TypeVarType
], "float or type variable">;

// Reference types (for reference operations)
def Polang_AnyRefType : AnyTypeOf<[
    Polang_MutRefType, Polang_RefType
], "any reference type">;

// Mutable reference or type variable
def Polang_MutRefOrTypeVar : AnyTypeOf<[
    Polang_MutRefType, Polang_TypeVarType
], "mutable reference or type variable">;

// Any reference or type variable
def Polang_AnyRefOrTypeVar : AnyTypeOf<[
    Polang_MutRefType, Polang_RefType, Polang_TypeVarType
], "any reference type or type variable">;

#endif // POLANG_TYPES_TD
