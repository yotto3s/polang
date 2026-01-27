//===- PolangASTTypes.td - Polang AST type definitions ------*- tablegen -*-===//
//
// Defines the types for the Polang AST dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_AST_TYPES_TD
#define POLANG_AST_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "polang/Dialect/PolangASTDialect.td"
include "polang/Dialect/PolangTypes.td"

//===----------------------------------------------------------------------===//
// Polang AST type definitions
//===----------------------------------------------------------------------===//

class PolangAST_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PolangAST_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Type variable for AST-level type inference
//===----------------------------------------------------------------------===//

def PolangAST_TypeVarType : PolangAST_Type<"TypeVar", "typevar"> {
  let summary = "AST-level type variable for type inference";
  let description = [{
    Represents an unresolved type at the AST level before type inference.
    Uses the same TypeVarKind enum as the main Polang dialect:
    - `any` (default): Can unify with any type
    - `integer`: Can only unify with integer types
    - `float`: Can only unify with float types

    Examples:
    ```mlir
    !polang_ast.typevar<1>          ; unconstrained type variable
    !polang_ast.typevar<2, integer> ; must resolve to an integer type
    !polang_ast.typevar<3, float>   ; must resolve to a float type
    ```
  }];

  let parameters = (ins
    "uint64_t":$id,
    DefaultValuedParameter<"::polang::TypeVarKind", "::polang::TypeVarKind::Any">:$kind
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool isAny() const { return getKind() == ::polang::TypeVarKind::Any; }
    bool isIntegerKind() const { return getKind() == ::polang::TypeVarKind::Integer; }
    bool isFloatKind() const { return getKind() == ::polang::TypeVarKind::Float; }
  }];
}

//===----------------------------------------------------------------------===//
// Type constraints for AST operations
//===----------------------------------------------------------------------===//

// Any type from the main Polang dialect or AST type variable
def PolangAST_AnyTypeOrVar : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_BoolType,
    Polang_TypeVarType, PolangAST_TypeVarType
], "any Polang type or type variable">;

// Numeric types or type variables
def PolangAST_AnyNumericOrVar : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType,
    Polang_TypeVarType, PolangAST_TypeVarType
], "Polang numeric type or type variable">;

// Bool or type variable (for conditions)
def PolangAST_BoolOrTypeVar : AnyTypeOf<[
    Polang_BoolType, Polang_TypeVarType, PolangAST_TypeVarType
], "bool or type variable">;

// Integer or type variable (for integer constants)
def PolangAST_IntegerOrTypeVar : AnyTypeOf<[
    Polang_IntegerType, Polang_TypeVarType, PolangAST_TypeVarType
], "integer or type variable">;

// Float or type variable (for float constants)
def PolangAST_FloatOrTypeVar : AnyTypeOf<[
    Polang_FloatType, Polang_TypeVarType, PolangAST_TypeVarType
], "float or type variable">;

#endif // POLANG_AST_TYPES_TD
