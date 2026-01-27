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
// CmpPredicate enum is defined in PolangEnums.td (shared with AST dialect)
//===----------------------------------------------------------------------===//

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
    For generic functions, type arguments can be provided.

    Examples:
    ```mlir
    // Regular function call
    %0 = polang.call @add(%a, %b) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>

    // Generic function call with type arguments
    %1 = polang.call @identity<[!polang.integer<64, signed>]>(%x) : (!polang.integer<64, signed>) -> !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    OptionalAttr<TypeArrayAttr>:$type_args,
    Variadic<Polang_AnyTypeOrVar>:$operands
  );
  let results = (outs Optional<Polang_AnyTypeOrVar>:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$callee, "mlir::TypeRange":$results,
                   CArg<"mlir::ValueRange", "{}">:$operands)>,
    OpBuilder<(ins "llvm::StringRef":$callee, "mlir::ArrayAttr":$typeArgs,
                   "mlir::TypeRange":$results,
                   CArg<"mlir::ValueRange", "{}">:$operands)>
  ];

  let extraClassDeclaration = [{
    mlir::FunctionType getCalleeType();

    /// Check if this is a generic function call (has type arguments).
    bool isGenericCall() { return getTypeArgs().has_value(); }
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
// Generic function operations
//===----------------------------------------------------------------------===//

def Polang_GenericFuncOp : Polang_Op<"generic_func", [
    IsolatedFromAbove,
    FunctionOpInterface,
    Symbol
  ]> {
  let summary = "Generic function definition with type parameters";
  let description = [{
    Defines a generic (polymorphic) function with explicit type parameters.
    Type parameters can have constraints (any, numeric, integer, float).

    Example:
    ```mlir
    polang.generic_func @identity<T>(%arg0: !polang.param<"T">) -> !polang.param<"T"> {
      polang.return %arg0 : !polang.param<"T">
    }

    polang.generic_func @add<T: numeric>(%a: !polang.param<"T", numeric>, %b: !polang.param<"T", numeric>) -> !polang.param<"T", numeric> {
      %r = polang.add %a, %b : !polang.param<"T", numeric>, !polang.param<"T", numeric> -> !polang.param<"T", numeric>
      polang.return %r : !polang.param<"T", numeric>
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    StrArrayAttr:$type_params,
    Polang_TypeParamKindArrayAttr:$constraints,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

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

    //===------------------------------------------------------------------===//
    // Type parameter methods
    //===------------------------------------------------------------------===//

    /// Get the number of type parameters.
    unsigned getNumTypeParams() {
      return getTypeParams().size();
    }

    /// Get the type parameter names as a vector of StringRef.
    llvm::SmallVector<llvm::StringRef> getTypeParamNames();

    /// Get the constraint for a type parameter by index.
    TypeParamKind getConstraintAt(unsigned index);

    /// Get the constraint for a type parameter by name.
    std::optional<TypeParamKind> getConstraintFor(llvm::StringRef name);

    /// Check if a name is a declared type parameter.
    bool hasTypeParam(llvm::StringRef name);
  }];
}

def Polang_SpecializedFuncOp : Polang_Op<"specialized_func", [Symbol]> {
  let summary = "Specialized (monomorphized) function reference";
  let description = [{
    References a generic function with concrete type arguments.
    This operation is used during monomorphization to record which
    concrete types a generic function should be specialized to.

    Example:
    ```mlir
    polang.specialized_func @identity$i64 from @identity<[!polang.integer<64, signed>]>
    polang.specialized_func @add$f64 from @add<[!polang.float<64>]>
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$generic_func,
    TypeArrayAttr:$type_args
  );

  let assemblyFormat = [{
    $sym_name `from` $generic_func `<` $type_args `>` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Get the specialized function type by substituting type arguments.
    mlir::FunctionType getSpecializedType(GenericFuncOp genericOp);
  }];
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
    ParentOneOf<["IfOp", "LetExprOp"]>
  ]> {
  let summary = "Yield value from region";
  let description = [{
    Yields a value from a region (if-then-else branches, let expressions).

    Example:
    ```mlir
    polang.yield %0 : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins Polang_AnyTypeOrVar:$value);

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
  let results = (outs Polang_AnyTypeOrVar:$result);
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
