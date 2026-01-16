//===- PolangOps.td - Polang operation definitions ---------*- tablegen -*-===//
//
// Defines the operations for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_OPS_TD
#define POLANG_OPS_TD

include "polang/Dialect/PolangDialect.td"
include "polang/Dialect/PolangTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Constant operations
//===----------------------------------------------------------------------===//

def Polang_ConstantIntOp : Polang_Op<"constant.int", [Pure]> {
  let summary = "Integer constant";
  let description = [{
    Produces a constant integer value.

    Example:
    ```mlir
    %0 = polang.constant.int 42 : !polang.int
    ```
  }];

  let arguments = (ins SI64Attr:$value);
  let results = (outs Polang_IntType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "int64_t":$value), [{
      build($_builder, $_state, $_builder.getType<IntType>(),
            $_builder.getIntegerAttr(
                $_builder.getIntegerType(64, /*isSigned=*/true), value));
    }]>
  ];
}

def Polang_ConstantDoubleOp : Polang_Op<"constant.double", [Pure]> {
  let summary = "Double constant";
  let description = [{
    Produces a constant double value.

    Example:
    ```mlir
    %0 = polang.constant.double 3.14 : !polang.double
    ```
  }];

  let arguments = (ins F64Attr:$value);
  let results = (outs Polang_DoubleType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "double":$value), [{
      build($_builder, $_state, $_builder.getType<DoubleType>(),
            $_builder.getF64FloatAttr(value));
    }]>
  ];
}

def Polang_ConstantBoolOp : Polang_Op<"constant.bool", [Pure]> {
  let summary = "Boolean constant";
  let description = [{
    Produces a constant boolean value.

    Example:
    ```mlir
    %0 = polang.constant.bool true : !polang.bool
    ```
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs Polang_BoolType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "bool":$value), [{
      build($_builder, $_state, $_builder.getType<BoolType>(),
            $_builder.getBoolAttr(value));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

class Polang_ArithOp<string mnemonic> : Polang_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins Polang_AnyNumericType:$lhs, Polang_AnyNumericType:$rhs);
  let results = (outs Polang_AnyNumericType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Polang_AddOp : Polang_ArithOp<"add"> {
  let summary = "Addition operation";
  let description = [{
    Adds two numeric values.

    Example:
    ```mlir
    %0 = polang.add %a, %b : !polang.int
    ```
  }];
}

def Polang_SubOp : Polang_ArithOp<"sub"> {
  let summary = "Subtraction operation";
  let description = [{
    Subtracts second operand from first.

    Example:
    ```mlir
    %0 = polang.sub %a, %b : !polang.int
    ```
  }];
}

def Polang_MulOp : Polang_ArithOp<"mul"> {
  let summary = "Multiplication operation";
  let description = [{
    Multiplies two numeric values.

    Example:
    ```mlir
    %0 = polang.mul %a, %b : !polang.int
    ```
  }];
}

def Polang_DivOp : Polang_ArithOp<"div"> {
  let summary = "Division operation";
  let description = [{
    Divides first operand by second.

    Example:
    ```mlir
    %0 = polang.div %a, %b : !polang.int
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Comparison operations
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

def Polang_CmpOp : Polang_Op<"cmp", [Pure, SameTypeOperands]> {
  let summary = "Comparison operation";
  let description = [{
    Compares two numeric values and returns a boolean result.

    Example:
    ```mlir
    %0 = polang.cmp eq, %a, %b : !polang.int
    ```
  }];

  let arguments = (ins
    Polang_CmpPredicate:$predicate,
    Polang_AnyNumericType:$lhs,
    Polang_AnyNumericType:$rhs
  );
  let results = (outs Polang_BoolType:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// Function operations
//===----------------------------------------------------------------------===//

def Polang_FuncOp : Polang_Op<"func", [
    IsolatedFromAbove,
    FunctionOpInterface,
    Symbol
  ]> {
  let summary = "Function definition";
  let description = [{
    Defines a function with a name, type signature, and body.

    Example:
    ```mlir
    polang.func @add(%a: !polang.int, %b: !polang.int) -> !polang.int {
      %c = polang.add %a, %b : !polang.int
      polang.return %c : !polang.int
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<StrArrayAttr>:$captures
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::FunctionType":$type,
                   CArg<"llvm::ArrayRef<llvm::StringRef>", "{}">:$captures)>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    mlir::Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Polang_CallOp : Polang_Op<"call", [
    DeclareOpInterfaceMethods<CallOpInterface>
  ]> {
  let summary = "Function call";
  let description = [{
    Calls a function with the given arguments.

    Example:
    ```mlir
    %0 = polang.call @add(%a, %b) : (!polang.int, !polang.int) -> !polang.int
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Polang_AnyType>:$operands);
  let results = (outs Optional<Polang_AnyType>:$result);

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$callee, "mlir::TypeRange":$results,
                   CArg<"mlir::ValueRange", "{}">:$operands)>
  ];

  let extraClassDeclaration = [{
    mlir::FunctionType getCalleeType();
  }];
}

def Polang_ReturnOp : Polang_Op<"return", [
    Pure,
    Terminator,
    ParentOneOf<["FuncOp"]>
  ]> {
  let summary = "Return from function";
  let description = [{
    Returns a value from a function.

    Example:
    ```mlir
    polang.return %0 : !polang.int
    ```
  }];

  let arguments = (ins Optional<Polang_AnyType>:$value);

  let assemblyFormat = "($value^ `:` type($value))? attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins), [{
      // Empty return - no operands
    }]>,
    OpBuilder<(ins "mlir::Value":$value), [{
      if (value)
        $_state.addOperands(value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

def Polang_IfOp : Polang_Op<"if", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "If-then-else expression";
  let description = [{
    Evaluates a condition and executes one of two regions.
    Both regions must yield a value of the same type.

    Example:
    ```mlir
    %0 = polang.if %cond -> !polang.int {
      polang.yield %a : !polang.int
    } else {
      polang.yield %b : !polang.int
    }
    ```
  }];

  let arguments = (ins Polang_BoolType:$condition);
  let results = (outs Polang_AnyType:$result);
  let regions = (region
    SizedRegion<1>:$thenRegion,
    SizedRegion<1>:$elseRegion
  );

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$condition)>
  ];
}

def Polang_YieldOp : Polang_Op<"yield", [
    Pure,
    Terminator,
    ParentOneOf<["IfOp"]>
  ]> {
  let summary = "Yield value from region";
  let description = [{
    Yields a value from a region (if-then-else branches).

    Example:
    ```mlir
    polang.yield %0 : !polang.int
    ```
  }];

  let arguments = (ins Polang_AnyType:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// Variable operations
//===----------------------------------------------------------------------===//

def Polang_AllocaOp : Polang_Op<"alloca"> {
  let summary = "Allocate stack memory for a variable";
  let description = [{
    Allocates memory on the stack for a variable.

    Example:
    ```mlir
    %0 = polang.alloca "x" : !polang.int -> memref<!polang.int>
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    TypeAttr:$elementType,
    UnitAttr:$isMutable
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "$name (`,` `mutable` $isMutable^)? `:` $elementType `->` type($result) attr-dict";
}

def Polang_LoadOp : Polang_Op<"load", [Pure]> {
  let summary = "Load value from memory";
  let description = [{
    Loads a value from a memory location.

    Example:
    ```mlir
    %0 = polang.load %ref : memref<i64> -> !polang.int
    ```
  }];

  let arguments = (ins AnyType:$ref);
  let results = (outs Polang_AnyType:$result);

  let assemblyFormat = "$ref `:` type($ref) `->` type($result) attr-dict";
}

def Polang_StoreOp : Polang_Op<"store"> {
  let summary = "Store value to memory";
  let description = [{
    Stores a value to a memory location.

    Example:
    ```mlir
    polang.store %value, %ref : !polang.int, memref<i64>
    ```
  }];

  let arguments = (ins Polang_AnyType:$value, AnyType:$ref);

  let assemblyFormat = "$value `,` $ref `:` type($value) `,` type($ref) attr-dict";
}

//===----------------------------------------------------------------------===//
// Print operation (for debugging/output)
//===----------------------------------------------------------------------===//

def Polang_PrintOp : Polang_Op<"print"> {
  let summary = "Print a value";
  let description = [{
    Prints a value to stdout. Used for debugging and output.

    Example:
    ```mlir
    polang.print %0 : !polang.int
    ```
  }];

  let arguments = (ins Polang_AnyType:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

#endif // POLANG_OPS_TD
