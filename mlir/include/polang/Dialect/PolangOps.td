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
// Attribute constraints
//===----------------------------------------------------------------------===//

// Generic integer attribute (any width, any signedness)
def Polang_AnyIntegerAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
                                  "integer attribute"> {
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::llvm::APInt }];
  let valueType = AnyInteger;
  let constBuilderCall = "::mlir::IntegerAttr()";
}

// Generic float attribute (any width)
def Polang_AnyFloatAttr : Attr<CPred<"::llvm::isa<::mlir::FloatAttr>($_self)">,
                                "floating-point attribute"> {
  let storageType = [{ ::mlir::FloatAttr }];
  let returnType = [{ ::llvm::APFloat }];
  let valueType = AnyFloat;
  let constBuilderCall = "::mlir::FloatAttr()";
}

//===----------------------------------------------------------------------===//
// Constant operations
//===----------------------------------------------------------------------===//

def Polang_ConstantIntegerOp : Polang_Op<"constant.integer", [Pure]> {
  let summary = "Integer constant";
  let description = [{
    Produces a constant integer value of any supported width and signedness.

    Example:
    ```mlir
    %0 = polang.constant.integer 42 : !polang.integer<64, signed>
    %1 = polang.constant.integer 255 : !polang.integer<8, unsigned>
    ```
  }];

  let arguments = (ins Polang_AnyIntegerAttr:$value);
  let results = (outs Polang_IntegerOrTypeVar:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "IntegerType":$type, "int64_t":$value), [{
      unsigned width = type.getWidth();
      bool isSigned = type.isSigned();
      build($_builder, $_state, type,
            $_builder.getIntegerAttr(
                $_builder.getIntegerType(width, isSigned), value));
    }]>,
    // Builder for type variables (before type inference)
    OpBuilder<(ins "mlir::Type":$type, "int64_t":$value), [{
      if (auto intType = ::llvm::dyn_cast<IntegerType>(type)) {
        unsigned width = intType.getWidth();
        bool isSigned = intType.isSigned();
        build($_builder, $_state, type,
              $_builder.getIntegerAttr(
                  $_builder.getIntegerType(width, isSigned), value));
      } else {
        // For type variables, use i64 attribute (actual type resolved later)
        build($_builder, $_state, type,
              $_builder.getIntegerAttr($_builder.getI64Type(), value));
      }
    }]>
  ];
}

def Polang_ConstantFloatOp : Polang_Op<"constant.float", [Pure]> {
  let summary = "Float constant";
  let description = [{
    Produces a constant float value of any supported width.

    Example:
    ```mlir
    %0 = polang.constant.float 3.14 : !polang.float<64>
    %1 = polang.constant.float 1.5 : !polang.float<32>
    ```
  }];

  let arguments = (ins Polang_AnyFloatAttr:$value);
  let results = (outs Polang_FloatOrTypeVar:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "FloatType":$type, "double":$value), [{
      unsigned width = type.getWidth();
      if (width == 32) {
        build($_builder, $_state, type,
              $_builder.getF32FloatAttr(static_cast<float>(value)));
      } else {
        build($_builder, $_state, type,
              $_builder.getF64FloatAttr(value));
      }
    }]>,
    // Builder for type variables (before type inference)
    OpBuilder<(ins "mlir::Type":$type, "double":$value), [{
      if (auto floatType = ::llvm::dyn_cast<FloatType>(type)) {
        unsigned width = floatType.getWidth();
        if (width == 32) {
          build($_builder, $_state, type,
                $_builder.getF32FloatAttr(static_cast<float>(value)));
        } else {
          build($_builder, $_state, type,
                $_builder.getF64FloatAttr(value));
        }
      } else {
        // For type variables, use f64 attribute (actual type resolved later)
        build($_builder, $_state, type, $_builder.getF64FloatAttr(value));
      }
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

class Polang_ArithOp<string mnemonic> : Polang_Op<mnemonic, [
    Pure
    // Note: SameOperandsAndResultType and InferTypeOpInterface removed
    // to allow type variables. Result type is specified explicitly in MLIRGen.
    // Custom verifier uses typesAreCompatible() to allow type vars.
  ]> {
  let arguments = (ins Polang_AnyNumericOrVar:$lhs, Polang_AnyNumericOrVar:$rhs);
  let results = (outs Polang_AnyNumericOrVar:$result);
  // Print all types since they may differ during type inference
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

  let hasVerifier = 1;
}

def Polang_AddOp : Polang_ArithOp<"add"> {
  let summary = "Addition operation";
  let description = [{
    Adds two numeric values.

    Example:
    ```mlir
    %0 = polang.add %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

def Polang_SubOp : Polang_ArithOp<"sub"> {
  let summary = "Subtraction operation";
  let description = [{
    Subtracts second operand from first.

    Example:
    ```mlir
    %0 = polang.sub %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

def Polang_MulOp : Polang_ArithOp<"mul"> {
  let summary = "Multiplication operation";
  let description = [{
    Multiplies two numeric values.

    Example:
    ```mlir
    %0 = polang.mul %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

def Polang_DivOp : Polang_ArithOp<"div"> {
  let summary = "Division operation";
  let description = [{
    Divides first operand by second.

    Example:
    ```mlir
    %0 = polang.div %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Type conversion operations
//===----------------------------------------------------------------------===//

def Polang_CastOp : Polang_Op<"cast", [Pure]> {
  let summary = "Explicit type conversion";
  let description = [{
    Converts a numeric value from one type to another.

    Supported conversions:
    - Integer to integer (widening/narrowing)
    - Float to float (widening/narrowing)
    - Integer to float
    - Float to integer (saturating)

    Example:
    ```mlir
    %0 = polang.cast %x : !polang.integer<32, signed> -> !polang.integer<64, signed>
    %1 = polang.cast %f : !polang.float<64> -> !polang.integer<32, signed>
    ```
  }];

  let arguments = (ins Polang_AnyNumericOrVar:$input);
  let results = (outs Polang_AnyNumericOrVar:$result);

  let assemblyFormat = "$input `:` type($input) `->` type($result) attr-dict";

  let hasVerifier = 1;
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

def Polang_CmpOp : Polang_Op<"cmp", [
    Pure,
    // Note: SameTypeOperands removed to allow type variables.
    // Custom verifier uses typesAreCompatible() to allow type vars.
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Comparison operation";
  let description = [{
    Compares two numeric values and returns a boolean result.

    Example:
    ```mlir
    %0 = polang.cmp eq, %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins
    Polang_CmpPredicate:$predicate,
    Polang_AnyNumericOrVar:$lhs,
    Polang_AnyNumericOrVar:$rhs
  );
  let results = (outs Polang_BoolType:$result);

  // Print both operand types since they may differ during type inference
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";

  let hasVerifier = 1;
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
    polang.func @add(%a: !polang.integer<64, signed>, %b: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
      %c = polang.add %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
      polang.return %c : !polang.integer<64, signed>
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
    DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Function call";
  let description = [{
    Calls a function with the given arguments.

    Example:
    ```mlir
    %0 = polang.call @add(%a, %b) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Polang_AnyTypeOrVar>:$operands);
  let results = (outs Optional<Polang_AnyTypeOrVar>:$result);

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

  let hasVerifier = 1;
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
    polang.return %0 : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins Optional<Polang_AnyTypeOrVar>:$value);

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

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

def Polang_IfOp : Polang_Op<"if", [
    RecursiveMemoryEffects
  ]> {
  let summary = "If-then-else expression";
  let description = [{
    Evaluates a condition and executes one of two regions.
    Both regions must yield a value of the same type.

    Example:
    ```mlir
    %0 = polang.if %cond : !polang.bool -> !polang.integer<64, signed> {
      polang.yield %a : !polang.integer<64, signed>
    } else {
      polang.yield %b : !polang.integer<64, signed>
    }
    ```
  }];

  let arguments = (ins Polang_BoolOrTypeVar:$condition);
  let results = (outs Polang_AnyTypeOrVar:$result);
  let regions = (region
    SizedRegion<1>:$thenRegion,
    SizedRegion<1>:$elseRegion
  );

  let assemblyFormat = [{
    $condition `:` type($condition) `->` type($result) $thenRegion `else` $elseRegion attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$condition)>
  ];

  let hasVerifier = 1;
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
    polang.yield %0 : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrVar:$value);

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
    %0 = polang.alloca "x" : !polang.integer<64, signed> -> memref<!polang.integer<64, signed>>
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
    %0 = polang.load %ref : memref<i64> -> !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins AnyType:$ref);
  let results = (outs Polang_AnyTypeOrVar:$result);

  let assemblyFormat = "$ref `:` type($ref) `->` type($result) attr-dict";
}

def Polang_StoreOp : Polang_Op<"store"> {
  let summary = "Store value to memory";
  let description = [{
    Stores a value to a memory location.

    Example:
    ```mlir
    polang.store %value, %ref : !polang.integer<64, signed>, memref<i64>
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrVar:$value, AnyType:$ref);

  let assemblyFormat = "$value `,` $ref `:` type($value) `,` type($ref) attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reference operations
//===----------------------------------------------------------------------===//

def Polang_MutRefDerefOp : Polang_Op<"mutref.deref", [Pure]> {
  let summary = "Dereference a mutable reference to read the value";
  let description = [{
    Reads the value from a mutable reference.

    Example:
    ```mlir
    %val = polang.mutref.deref %ref : !polang.mutref<!polang.integer<64, signed>> -> !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins Polang_MutRefOrTypeVar:$ref);
  let results = (outs Polang_AnyTypeOrVar:$result);

  let assemblyFormat = "$ref `:` type($ref) `->` type($result) attr-dict";
}

def Polang_MutRefStoreOp : Polang_Op<"mutref.store"> {
  let summary = "Store a value through a mutable reference";
  let description = [{
    Writes a value through a mutable reference.

    Example:
    ```mlir
    polang.mutref.store %val, %ref : !polang.integer<64, signed>, !polang.mutref<!polang.integer<64, signed>>
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrVar:$value, Polang_MutRefOrTypeVar:$ref);
  let results = (outs Polang_AnyTypeOrVar:$result);

  let assemblyFormat = "$value `,` $ref `:` type($value) `,` type($ref) `->` type($result) attr-dict";
}

def Polang_RefCreateOp : Polang_Op<"ref.create", [Pure]> {
  let summary = "Create an immutable reference from a mutable reference";
  let description = [{
    Creates an immutable reference from a mutable reference.
    The resulting reference can only be used for reading.

    Example:
    ```mlir
    %imm = polang.ref.create %mut : !polang.mutref<!polang.integer<64, signed>> -> !polang.ref<!polang.integer<64, signed>>
    ```
  }];

  let arguments = (ins Polang_MutRefOrTypeVar:$source);
  let results = (outs Polang_AnyRefOrTypeVar:$result);

  let assemblyFormat = "$source `:` type($source) `->` type($result) attr-dict";
}

def Polang_RefDerefOp : Polang_Op<"ref.deref", [Pure]> {
  let summary = "Dereference an immutable reference to read the value";
  let description = [{
    Reads the value from an immutable reference.

    Example:
    ```mlir
    %val = polang.ref.deref %ref : !polang.ref<!polang.integer<64, signed>> -> !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins AnyTypeOf<[Polang_RefType, Polang_TypeVarType]>:$ref);
  let results = (outs Polang_AnyTypeOrVar:$result);

  let assemblyFormat = "$ref `:` type($ref) `->` type($result) attr-dict";
}

def Polang_MutRefCreateOp : Polang_Op<"mutref.create"> {
  let summary = "Create a mutable reference from an initial value";
  let description = [{
    Creates a new mutable reference initialized with the given value.
    This allocates memory for the value and returns a mutable reference.

    Example:
    ```mlir
    %ref = polang.mutref.create %val : !polang.integer<64, signed> -> !polang.mutref<!polang.integer<64, signed>>
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrVar:$initialValue);
  let results = (outs Polang_MutRefOrTypeVar:$result);

  let assemblyFormat = "$initialValue `:` type($initialValue) `->` type($result) attr-dict";
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
    polang.print %0 : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrVar:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

#endif // POLANG_OPS_TD
