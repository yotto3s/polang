//===- PolangOps.td - Polang operation definitions ---------*- tablegen -*-===//
//
// Defines the operations for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_OPS_TD
#define POLANG_OPS_TD

include "polang/Dialect/PolangDialect.td"
include "polang/Dialect/PolangTypes.td"
include "polang/Dialect/PolangEnums.td"
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
    Produces a constant integer or index value.

    Example:
    ```mlir
    %0 = polang.constant.integer 42 : i64
    %1 = polang.constant.integer 255 : i8
    ```
  }];

  let arguments = (ins Polang_AnyIntegerAttr:$value);
  let results = (outs AnyTypeOf<[Polang_NonBoolInteger, Index, Polang_TypeParamType]>:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::Type":$type, "int64_t":$value), [{
      if (auto intType = ::llvm::dyn_cast<::mlir::IntegerType>(type)) {
        build($_builder, $_state, type,
              $_builder.getIntegerAttr(intType, value));
      } else {
        // For index types and type parameters, use i64 attribute
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
    %0 = polang.constant.float 3.14 : f64
    %1 = polang.constant.float 1.5 : f32
    ```
  }];

  let arguments = (ins Polang_AnyFloatAttr:$value);
  let results = (outs AnyTypeOf<[AnyFloat, Polang_TypeParamType]>:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::Type":$type, "double":$value), [{
      if (auto floatType = ::llvm::dyn_cast<::mlir::FloatType>(type)) {
        build($_builder, $_state, type,
              $_builder.getFloatAttr(floatType, value));
      } else {
        // For type parameters, use f64 attribute (actual type resolved later)
        build($_builder, $_state, type, $_builder.getF64FloatAttr(value));
      }
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
  let arguments = (ins Polang_AnyNumericOrParam:$lhs, Polang_AnyNumericOrParam:$rhs);
  let results = (outs Polang_AnyNumericOrParam:$result);
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

def Polang_RemOp : Polang_ArithOp<"rem"> {
  let summary = "Remainder (modulo) operation";
  let description = [{
    Computes the remainder of integer division (truncated division).
    The result has the same sign as the dividend.
    Only valid for integer and index types; other types (float, bool, etc.) are not permitted.

    Example:
    ```mlir
    %0 = polang.rem %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Unary operations
//===----------------------------------------------------------------------===//

def Polang_NegOp : Polang_Op<"neg", [Pure]> {
  let summary = "Unary negation operation";
  let description = [{
    Negates a numeric value (unary minus).

    Example:
    ```mlir
    %0 = polang.neg %a : !polang.integer<64, signed> -> !polang.integer<64, signed>
    %1 = polang.neg %f : !polang.float<64> -> !polang.float<64>
    ```
  }];

  let arguments = (ins Polang_AnyNumericOrParam:$operand);
  let results = (outs Polang_AnyNumericOrParam:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";

  let hasVerifier = 1;
}

def Polang_NotOp : Polang_Op<"not", [
    Pure,
    SameOperandsAndResultType
]> {
  let summary = "Logical not operation";
  let description = [{
    Performs logical not on an i1 (boolean) value.

    Example:
    ```mlir
    %0 = polang.not %a : i1
    ```
  }];

  let arguments = (ins I1:$operand);
  let results = (outs I1:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
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
    %0 = polang.cast %x : i32 -> i64
    %1 = polang.cast %f : f64 -> i32
    ```
  }];

  let arguments = (ins Polang_AnyNumericOrParam:$input);
  let results = (outs Polang_AnyNumericOrParam:$result);

  let assemblyFormat = "$input `:` type($input) `->` type($result) attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Comparison operations
// CmpPredicate enum is defined in PolangEnums.td
//===----------------------------------------------------------------------===//

def Polang_CmpOp : Polang_Op<"cmp", [
    Pure,
    // Note: SameTypeOperands removed to allow type variables.
    // Custom verifier uses typesAreCompatible() to allow type vars.
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Comparison operation";
  let description = [{
    Compares two numeric values and returns an i1 boolean result.

    Example:
    ```mlir
    %0 = polang.cmp eq, %a, %b : i64, i64
    ```
  }];

  let arguments = (ins
    Polang_CmpPredicate:$predicate,
    Polang_AnyNumericOrParam:$lhs,
    Polang_AnyNumericOrParam:$rhs
  );
  let results = (outs I1:$result);

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

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Polang_AnyTypeOrParam>:$operands);
  let results = (outs Optional<Polang_AnyTypeOrParam>:$result);

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

def Polang_GenericFuncOp : Polang_Op<"generic_func", [
    IsolatedFromAbove,
    FunctionOpInterface,
    Symbol
  ]> {
  let summary = "Generic function definition with type parameters";
  let description = [{
    Defines a polymorphic function with named type parameters and optional
    trait bounds. Type parameters use ML-style naming ('a, 'b, etc.).

    Example:
    ```mlir
    polang.generic_func @add<'a: Numeric>(%x: !polang.type_param<"a">, %y: !polang.type_param<"a">)
        -> !polang.type_param<"a"> {
      %0 = polang.add %x, %y : !polang.type_param<"a">, !polang.type_param<"a"> -> !polang.type_param<"a">
      polang.return %0 : !polang.type_param<"a">
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    StrArrayAttr:$type_params,
    OptionalAttr<StrArrayAttr>:$type_param_bounds,
    OptionalAttr<StrArrayAttr>:$captures
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::FunctionType":$type,
                   "llvm::ArrayRef<llvm::StringRef>":$typeParams,
                   CArg<"llvm::ArrayRef<llvm::StringRef>", "{}">:$typeParamBounds,
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

def Polang_InstantiateOp : Polang_Op<"instantiate", [
    DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Instantiate a generic function with concrete types";
  let description = [{
    Calls a generic function with concrete type bindings for its type parameters.

    Example:
    ```mlir
    %0 = polang.instantiate @identity<'a = !polang.integer<64, signed>>(%arg)
        : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    StrArrayAttr:$type_param_names,
    TypeArrayAttr:$type_bindings,
    Variadic<Polang_AnyTypeOrParam>:$operands
  );
  let results = (outs Optional<Polang_AnyTypeOrParam>:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$callee, "mlir::TypeRange":$results,
                   "mlir::ValueRange":$operands,
                   "llvm::ArrayRef<llvm::StringRef>":$typeParamNames,
                   "llvm::ArrayRef<mlir::Type>":$typeBindings)>
  ];

  let extraClassDeclaration = [{
    mlir::FunctionType getCalleeType();
  }];

  let hasVerifier = 1;
}

def Polang_ReturnOp : Polang_Op<"return", [
    Pure,
    Terminator,
    ParentOneOf<["FuncOp", "GenericFuncOp"]>
  ]> {
  let summary = "Return from function";
  let description = [{
    Returns a value from a function.

    Example:
    ```mlir
    polang.return %0 : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins Optional<Polang_AnyTypeOrParam>:$value);

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

def Polang_YieldOp : Polang_Op<"yield", [
    Pure,
    Terminator,
    HasParent<"LetExprOp">
  ]> {
  let summary = "Yield value from region";
  let description = [{
    Yields a value from a let expression body region.

    Example:
    ```mlir
    polang.yield %0 : i64
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrParam:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// Let expression operations
//===----------------------------------------------------------------------===//

def Polang_LetExprOp : Polang_Op<"let_expr", [RecursiveMemoryEffects]> {
  let summary = "Let-in expression with separate binding regions";
  let description = [{
    A let expression with separate regions for each binding and a body.
    Each binding region computes a value independently and yields it via
    yield.binding. The body region receives bound values as block arguments.

    Binding regions are isolated from each other, enabling parallel evaluation.

    Example:
    ```mlir
    %result = polang.let_expr ["x", "y"] -> !polang.integer<64, signed>
      binding {
        polang.yield.binding %val1 : !polang.integer<64, signed>
      }
      binding {
        polang.yield.binding %val2 : !polang.integer<64, signed>
      }
      body {
      ^bb0(%x: !polang.integer<64, signed>, %y: !polang.integer<64, signed>):
        %sum = polang.add %x, %y : ...
        polang.yield %sum : !polang.integer<64, signed>
      }
    ```
  }];

  let arguments = (ins StrArrayAttr:$var_names);
  let results = (outs Polang_AnyTypeOrParam:$result);
  // Body must come first since MLIR requires variadic regions to be last
  let regions = (region
    SizedRegion<1>:$body,
    VariadicRegion<SizedRegion<1>>:$bindings
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Polang_YieldBindingOp : Polang_Op<"yield.binding", [
    Pure,
    Terminator,
    HasParent<"LetExprOp">
  ]> {
  let summary = "Yield bound value from a binding region";
  let description = [{
    Terminates a binding region of a let expression and yields
    the bound value as input to the body region's corresponding block argument.

    Example:
    ```mlir
    polang.yield.binding %x : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrParam:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// Variable operations
//===----------------------------------------------------------------------===//

def Polang_GlobalOp : Polang_Op<"global", [Symbol]> {
  let summary = "Global variable declaration";
  let description = [{
    Declares a module-level global variable with a name and type.
    Used in the REPL for cross-module variable persistence.

    When `is_external` is set, the global is an extern declaration
    (defined in a previously compiled module; JIT resolves the symbol).
    Otherwise, this is a definition. Non-external globals may have an
    optional initializer region that computes the initial value.

    Example:
    ```mlir
    polang.global @x : !polang.integer<64, signed> {
      %v = polang.constant 42 : !polang.integer<64, signed>
      polang.yield.global %v : !polang.integer<64, signed>
    }
    polang.global @y {is_external} : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_external
  );

  let regions = (region MaxSizedRegion<1>:$initializer);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Polang_YieldGlobalOp : Polang_Op<"yield.global", [
    Pure,
    Terminator,
    HasParent<"GlobalOp">
  ]> {
  let summary = "Yield initial value from a global initializer region";
  let description = [{
    Terminates the initializer region of a GlobalOp and yields
    the initial value for the global variable.

    Example:
    ```mlir
    polang.yield.global %v : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrParam:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

def Polang_GlobalLoadOp : Polang_Op<"global.load", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    Pure
  ]> {
  let summary = "Load a value from a global variable";
  let description = [{
    Loads the current value from a previously declared global variable.

    Example:
    ```mlir
    %v = polang.global.load @x : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs Polang_AnyTypeOrParam:$result);

  let assemblyFormat = "$global_name `:` type($result) attr-dict";

  let hasVerifier = 1;
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

  let arguments = (ins Polang_AnyTypeOrParam:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

#endif // POLANG_OPS_TD
