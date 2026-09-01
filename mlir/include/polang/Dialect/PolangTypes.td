//===- PolangTypes.td - Polang type definitions ------------*- tablegen -*-===//
//
// Defines the types for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_TYPES_TD
#define POLANG_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
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

def Polang_IntegerType : Polang_Type<"Integer", "integer", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
]> {
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
    uint64_t typeSize() const { return (getWidth() + 7) / 8; }
  }];
}

def Polang_FloatType : Polang_Type<"Float", "float", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
]> {
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

  let extraClassDeclaration = [{
    uint64_t typeSize() const { return getWidth() / 8; }
  }];
}

def Polang_BoolType : Polang_Type<"Bool", "bool", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
]> {
  let summary = "1-bit boolean type";
  let description = [{
    The Polang boolean type represents a true/false value.
    It maps to `bool` in Polang source code.
  }];

  let extraClassDeclaration = [{
    uint64_t typeSize() const { return 1; }
  }];
}

def Polang_IndexType : Polang_Type<"Index", "index", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
]> {
  let summary = "Platform-native index type with signedness";
  let description = [{
    Represents isize (signed) and usize (unsigned) types that map to
    the platform-native pointer width. Both lower to MLIR's built-in
    index type, with signedness tracked for correct cast lowering.

    Examples:
    ```mlir
    !polang.index<signed>    ; isize
    !polang.index<unsigned>  ; usize
    ```
  }];

  let parameters = (ins "Signedness":$signedness);
  let assemblyFormat = "`<` $signedness `>`";

  let extraClassDeclaration = [{
    bool isSigned() const { return getSignedness() == Signedness::Signed; }
    bool isUnsigned() const { return getSignedness() == Signedness::Unsigned; }
    uint64_t typeSize(const ::mlir::DataLayout *dataLayout = nullptr) const;
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

def Polang_TupleType : Polang_Type<"Tuple", "tuple", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
]> {
  let summary = "Tuple type containing zero or more types";
  let description = [{
    Represents a fixed-size heterogeneous product of types.
    Can contain any Polang types or type parameters.

    Examples:
    ```mlir
    !polang.tuple<>                                   ; 0-tuple (unit)
    !polang.tuple<!polang.index<unsigned>>            ; 1-tuple
    !polang.tuple<!polang.index<unsigned>, !polang.index<unsigned>> ; 2-tuple
    !polang.tuple<!polang.integer<64, signed>, !polang.tuple<!polang.float<64>, !polang.bool>> ; nested tuple
    !polang.tuple<!polang.integer<64, signed>, !polang.type_param<"a">>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$types);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Size in bytes (computed with 64-bit member alignment), or std::nullopt
    /// if the tuple contains unsized/generic types (e.g. TypeParamType).
    ::std::optional<uint64_t> typeSize(const ::mlir::DataLayout *dataLayout = nullptr) const;

    /// Number of 64-bit words (slots) occupied by this tuple, or std::nullopt
    /// if unspecialized.
    ::std::optional<uint64_t> getNumSlots(const ::mlir::DataLayout *dataLayout = nullptr) const {
      auto sz = typeSize(dataLayout);
      if (!sz) return ::std::nullopt;
      return (*sz + 7) / 8;
    }

    /// Offset in bytes of the element at the specified index, or std::nullopt
    /// if any preceding element has unknown size.
    ::std::optional<uint64_t> getElementOffset(size_t index, const ::mlir::DataLayout *dataLayout = nullptr) const;

    /// Offset in 64-bit slots of the element at the specified index, or std::nullopt
    /// if any preceding element has unknown size.
    ::std::optional<uint64_t> getElementSlotOffset(size_t index, const ::mlir::DataLayout *dataLayout = nullptr) const {
      auto off = getElementOffset(index, dataLayout);
      if (!off) return ::std::nullopt;
      return *off / 8;
    }

    /// Offsets in bytes of all elements, or std::nullopt if any element has
    /// unknown size.
    ::std::optional<::llvm::SmallVector<uint64_t>> getElementOffsets(const ::mlir::DataLayout *dataLayout = nullptr) const;

    /// Offsets in 64-bit slots of all elements, or std::nullopt if any element
    /// has unknown size.
    ::std::optional<::llvm::SmallVector<uint64_t>> getElementSlotOffsets(const ::mlir::DataLayout *dataLayout = nullptr) const;
  }];
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def Polang_AnyNumericType : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_IndexType
], "Polang numeric type">;

def Polang_AnyType : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_BoolType, Polang_IndexType, Polang_TupleType
], "any Polang type">;

// Type constraints that include type parameters (for generic functions)
def Polang_AnyNumericOrParam : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_IndexType, Polang_TypeParamType
], "Polang numeric type or type parameter">;

def Polang_AnyTypeOrParam : AnyTypeOf<[
    Polang_IntegerType, Polang_FloatType, Polang_BoolType, Polang_IndexType, Polang_TypeParamType, Polang_TupleType
], "any Polang type or type parameter">;

#endif // POLANG_TYPES_TD
