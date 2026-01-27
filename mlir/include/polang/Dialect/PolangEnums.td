//===- PolangEnums.td - Polang enum definitions -----*- tablegen -*-===//
//
// Enum definitions shared between Polang and PolangAST dialects.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_ENUMS_TD
#define POLANG_ENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Comparison predicate enum
//===----------------------------------------------------------------------===//

def Polang_CmpPredicateEQ  : I64EnumAttrCase<"eq", 0>;
def Polang_CmpPredicateNE  : I64EnumAttrCase<"ne", 1>;
def Polang_CmpPredicateLT  : I64EnumAttrCase<"lt", 2>;
def Polang_CmpPredicateLE  : I64EnumAttrCase<"le", 3>;
def Polang_CmpPredicateGT  : I64EnumAttrCase<"gt", 4>;
def Polang_CmpPredicateGE  : I64EnumAttrCase<"ge", 5>;

def Polang_CmpPredicate : I64EnumAttr<
    "CmpPredicate", "comparison predicate",
    [Polang_CmpPredicateEQ, Polang_CmpPredicateNE,
     Polang_CmpPredicateLT, Polang_CmpPredicateLE,
     Polang_CmpPredicateGT, Polang_CmpPredicateGE]> {
  let cppNamespace = "::polang";
}

#endif // POLANG_ENUMS_TD
