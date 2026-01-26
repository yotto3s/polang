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
// TypeParamKind enum for explicit generic type parameter constraints
//===----------------------------------------------------------------------===//

def Polang_TypeParamKind_Any : I32EnumAttrCase<"Any", 0, "any">;
def Polang_TypeParamKind_Numeric : I32EnumAttrCase<"Numeric", 1, "numeric">;
def Polang_TypeParamKind_Integer : I32EnumAttrCase<"Integer", 2, "integer">;
def Polang_TypeParamKind_Float : I32EnumAttrCase<"Float", 3, "float">;

def Polang_TypeParamKind : I32EnumAttr<"TypeParamKind",
    "type parameter constraint kind", [
    Polang_TypeParamKind_Any,
    Polang_TypeParamKind_Numeric,
    Polang_TypeParamKind_Integer,
    Polang_TypeParamKind_Float
]> {
  let cppNamespace = "::polang";
}

def Polang_TypeParamKindAttr : EnumAttr<Polang_Dialect, Polang_TypeParamKind,
    "type_param_kind">;

def Polang_TypeParamKindArrayAttr : TypedArrayAttrBase<Polang_TypeParamKindAttr,
    "array of type parameter constraint kinds">;

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

def Polang_TypeParamType : Polang_Type<"TypeParam", "param"> {
  let summary = "Named type parameter for explicit generics";
  let description = [{
    Represents a named type parameter in a generic function definition.
    Unlike TypeVarType (which uses numeric IDs for type inference),
    TypeParamType uses string names matching the source-level syntax.

    The optional `constraint` parameter restricts what concrete types
    can be substituted:
    - `any` (default): Any type is allowed
    - `numeric`: Only integer or float types
    - `integer`: Only integer types
    - `float`: Only float types

    Examples:
    ```mlir
    !polang.param<"T">           ; unconstrained type parameter
    !polang.param<"N", numeric>  ; must be integer or float
    !polang.param<"I", integer>  ; must be integer
    !polang.param<"F", float>    ; must be float
    ```
  }];

  let parameters = (ins
    StringRefParameter<"name">:$name,
    DefaultValuedParameter<"TypeParamKind", "TypeParamKind::Any">:$constraint
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Check if a concrete type satisfies this parameter's constraint.
    bool satisfiesConstraint(mlir::Type concreteType) const;

    bool isAny() const { return getConstraint() == TypeParamKind::Any; }
    bool isNumeric() const { return getConstraint() == TypeParamKind::Numeric; }
    bool isIntegerKind() const { return getConstraint() == TypeParamKind::Integer; }
    bool isFloatKind() const { return getConstraint() == TypeParamKind::Float; }
  }];
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def Polang_AnyNumericType : AnyTypeOf<[Polang_IntegerType, Polang_FloatType],
    "Polang numeric type">;

def Polang_AnyType : AnyTypeOf<[Polang_IntegerType, Polang_FloatType, Polang_BoolType],
    "any Polang type">;

// Type constraints that include type variables and type parameters
// (for use before type inference or specialization)
def Polang_AnyNumericOrVar : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_TypeVarType, Polang_TypeParamType
], "Polang numeric type, type variable, or type parameter">;

def Polang_AnyTypeOrVar : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_BoolType,
    Polang_TypeVarType, Polang_TypeParamType
], "any Polang type, type variable, or type parameter">;

// Bool or type variable/parameter (for if conditions before type inference)
def Polang_BoolOrTypeVar : AnyTypeOf<[
    Polang_BoolType, Polang_TypeVarType, Polang_TypeParamType
], "bool, type variable, or type parameter">;

// Integer or type variable/parameter (for integer constants before type inference)
def Polang_IntegerOrTypeVar : AnyTypeOf<[
    Polang_IntegerType, Polang_TypeVarType, Polang_TypeParamType
], "integer, type variable, or type parameter">;

// Float or type variable/parameter (for float constants before type inference)
def Polang_FloatOrTypeVar : AnyTypeOf<[
    Polang_FloatType, Polang_TypeVarType, Polang_TypeParamType
], "float, type variable, or type parameter">;

//===----------------------------------------------------------------------===//
// Type constraints that include type parameters (for explicit generics)
//===----------------------------------------------------------------------===//

// Numeric or type parameter (for generic numeric operations)
def Polang_NumericOrParam : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_TypeParamType
], "Polang numeric type or type parameter">;

// Any Polang type or type parameter
def Polang_AnyOrParam : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_BoolType, Polang_TypeParamType
], "any Polang type or type parameter">;

// Bool or type parameter
def Polang_BoolOrParam : AnyTypeOf<[
    Polang_BoolType, Polang_TypeParamType
], "bool or type parameter">;

// Integer or type parameter
def Polang_IntegerOrParam : AnyTypeOf<[
    Polang_IntegerType, Polang_TypeParamType
], "integer or type parameter">;

// Float or type parameter
def Polang_FloatOrParam : AnyTypeOf<[
    Polang_FloatType, Polang_TypeParamType
], "float or type parameter">;

#endif // POLANG_TYPES_TD
