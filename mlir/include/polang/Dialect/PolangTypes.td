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

def Polang_TupleType : Polang_Type<"Tuple", "tuple", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
]> {
  let summary = "Tuple type containing zero or more types";
  let description = [{
    Represents a fixed-size heterogeneous product of types.
    Can contain any MLIR standard types, Polang tuple types, or type parameters.

    Examples:
    ```mlir
    !polang.tuple<>                          ; 0-tuple (unit)
    !polang.tuple<i64, f64>                  ; 2-tuple of i64 and f64
    !polang.tuple<i64, !polang.tuple<f64, i1>> ; nested tuple
    !polang.tuple<i64, !polang.type_param<"a">>
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

def Polang_AnyNumericOrParam : AnyTypeOf<[
    AnyInteger, AnyFloat, Index, Polang_TypeParamType
], "numeric type or type parameter">;

def Polang_AnyTypeOrParam : AnyTypeOf<[
    AnyInteger, AnyFloat, Index, Polang_TupleType, Polang_TypeParamType
], "any type or type parameter">;

def Polang_I1OrParamType : AnyTypeOf<[
    I1, Polang_TypeParamType
], "i1 (bool) or type parameter">;

#endif // POLANG_TYPES_TD
