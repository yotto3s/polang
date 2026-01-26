//===- PolangASTOps.td - Polang AST operation definitions ---*- tablegen -*-===//
//
// Defines the operations for the Polang AST dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POLANG_AST_OPS_TD
#define POLANG_AST_OPS_TD

include "polang/Dialect/PolangASTDialect.td"
include "polang/Dialect/PolangASTTypes.td"
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
def PolangAST_AnyIntegerAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
                                  "integer attribute"> {
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::llvm::APInt }];
  let valueType = AnyInteger;
  let constBuilderCall = "::mlir::IntegerAttr()";
}

// Generic float attribute (any width)
def PolangAST_AnyFloatAttr : Attr<CPred<"::llvm::isa<::mlir::FloatAttr>($_self)">,
                                "floating-point attribute"> {
  let storageType = [{ ::mlir::FloatAttr }];
  let returnType = [{ ::llvm::APFloat }];
  let valueType = AnyFloat;
  let constBuilderCall = "::mlir::FloatAttr()";
}

//===----------------------------------------------------------------------===//
// Constant operations
//===----------------------------------------------------------------------===//

def PolangAST_ConstantIntegerOp : PolangAST_Op<"constant.integer", [Pure]> {
  let summary = "AST-level integer constant";
  let description = [{
    Produces a constant integer value. Result type can be a concrete integer
    type or a type variable for type inference.

    Example:
    ```mlir
    %0 = polang_ast.constant.integer 42 : !polang.integer<64, signed>
    %1 = polang_ast.constant.integer 42 : !polang_ast.typevar<1, integer>
    ```
  }];

  let arguments = (ins PolangAST_AnyIntegerAttr:$value);
  let results = (outs PolangAST_IntegerOrTypeVar:$result);

  let hasCustomAssemblyFormat = 1;
}

def PolangAST_ConstantFloatOp : PolangAST_Op<"constant.float", [Pure]> {
  let summary = "AST-level float constant";
  let description = [{
    Produces a constant float value. Result type can be a concrete float
    type or a type variable for type inference.

    Example:
    ```mlir
    %0 = polang_ast.constant.float 3.14 : !polang.float<64>
    %1 = polang_ast.constant.float 3.14 : !polang_ast.typevar<1, float>
    ```
  }];

  let arguments = (ins PolangAST_AnyFloatAttr:$value);
  let results = (outs PolangAST_FloatOrTypeVar:$result);

  let hasCustomAssemblyFormat = 1;
}

def PolangAST_ConstantBoolOp : PolangAST_Op<"constant.bool", [Pure]> {
  let summary = "AST-level boolean constant";
  let description = [{
    Produces a constant boolean value.

    Example:
    ```mlir
    %0 = polang_ast.constant.bool true : !polang.bool
    ```
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs Polang_BoolType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

class PolangAST_ArithOp<string mnemonic> : PolangAST_Op<mnemonic, [Pure]> {
  let arguments = (ins PolangAST_AnyNumericOrVar:$lhs, PolangAST_AnyNumericOrVar:$rhs);
  let results = (outs PolangAST_AnyNumericOrVar:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

  let hasVerifier = 1;
}

def PolangAST_AddOp : PolangAST_ArithOp<"add"> {
  let summary = "AST-level addition operation";
  let description = [{
    Adds two numeric values. Types may include type variables.

    Example:
    ```mlir
    %0 = polang_ast.add %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

def PolangAST_SubOp : PolangAST_ArithOp<"sub"> {
  let summary = "AST-level subtraction operation";
  let description = [{
    Subtracts second operand from first. Types may include type variables.

    Example:
    ```mlir
    %0 = polang_ast.sub %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

def PolangAST_MulOp : PolangAST_ArithOp<"mul"> {
  let summary = "AST-level multiplication operation";
  let description = [{
    Multiplies two numeric values. Types may include type variables.

    Example:
    ```mlir
    %0 = polang_ast.mul %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

def PolangAST_DivOp : PolangAST_ArithOp<"div"> {
  let summary = "AST-level division operation";
  let description = [{
    Divides first operand by second. Types may include type variables.

    Example:
    ```mlir
    %0 = polang_ast.div %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Type conversion operations
//===----------------------------------------------------------------------===//

def PolangAST_CastOp : PolangAST_Op<"cast", [Pure]> {
  let summary = "AST-level type conversion";
  let description = [{
    Converts a numeric value from one type to another.
    Input and output types may be type variables.

    Example:
    ```mlir
    %0 = polang_ast.cast %x : !polang.integer<32, signed> -> !polang.float<64>
    ```
  }];

  let arguments = (ins PolangAST_AnyNumericOrVar:$input);
  let results = (outs PolangAST_AnyNumericOrVar:$result);

  let assemblyFormat = "$input `:` type($input) `->` type($result) attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

// Reuse CmpPredicate enum from the main polang dialect (defined in PolangOps.td)

def PolangAST_CmpOp : PolangAST_Op<"cmp", [Pure]> {
  let summary = "AST-level comparison operation";
  let description = [{
    Compares two numeric values and returns a boolean result.
    Operand types may be type variables.

    Example:
    ```mlir
    %0 = polang_ast.cmp eq, %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins
    Polang_CmpPredicate:$predicate,
    PolangAST_AnyNumericOrVar:$lhs,
    PolangAST_AnyNumericOrVar:$rhs
  );
  let results = (outs Polang_BoolType:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

def PolangAST_IfOp : PolangAST_Op<"if", [RecursiveMemoryEffects]> {
  let summary = "AST-level if-then-else expression";
  let description = [{
    Evaluates a condition and executes one of two regions.
    Both regions must yield a value of compatible type.

    Example:
    ```mlir
    %0 = polang_ast.if %cond : !polang.bool -> !polang.integer<64, signed> {
      polang_ast.yield %a : !polang.integer<64, signed>
    } else {
      polang_ast.yield %b : !polang.integer<64, signed>
    }
    ```
  }];

  let arguments = (ins PolangAST_BoolOrTypeVar:$condition);
  let results = (outs PolangAST_AnyTypeOrVar:$result);
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

def PolangAST_YieldOp : PolangAST_Op<"yield", [Pure, Terminator, ParentOneOf<["IfOp", "LetExprOp"]>]> {
  let summary = "AST-level yield from region";
  let description = [{
    Yields a value from a region (if-then-else branches, let expressions).

    Example:
    ```mlir
    polang_ast.yield %0 : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins PolangAST_AnyTypeOrVar:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

def PolangAST_YieldBindingOp : PolangAST_Op<"yield.binding",
    [Pure, Terminator, HasParent<"LetExprOp">]> {
  let summary = "Yield bound value from a binding region";
  let description = [{
    Terminates a binding region of a let expression and yields
    the bound value as input to the body region's corresponding block argument.

    Example:
    ```mlir
    polang_ast.yield.binding %x : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins PolangAST_AnyTypeOrVar:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// Function operations
//===----------------------------------------------------------------------===//

def PolangAST_FuncOp : PolangAST_Op<"func", [
    FunctionOpInterface,
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "AST-level function declaration";
  let description = [{
    Declares a function. Parameter and return types may include type variables
    for type inference.

    Example:
    ```mlir
    polang_ast.func @add(%a: !polang.integer<64, signed>, %b: !polang.integer<64, signed>) -> !polang.integer<64, signed> {
      %0 = polang_ast.add %a, %b : !polang.integer<64, signed>, !polang.integer<64, signed> -> !polang.integer<64, signed>
      polang_ast.return %0 : !polang.integer<64, signed>
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "::mlir::StringRef":$name, "::mlir::FunctionType":$type,
                   CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs,
                   CArg<"::mlir::ArrayRef<::mlir::DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Returns the callable region for this function.
    ::mlir::Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PolangAST_CallOp : PolangAST_Op<"call", [
    DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "AST-level function call";
  let description = [{
    Calls a function by name with the given arguments.

    Example:
    ```mlir
    %0 = polang_ast.call @add(%a, %b) : (!polang.integer<64, signed>, !polang.integer<64, signed>) -> !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];

  let extraClassDeclaration = [{
    /// Return the callee type for this call.
    ::mlir::FunctionType getCalleeType();
  }];
}

def PolangAST_ReturnOp : PolangAST_Op<"return", [Pure, Terminator, HasParent<"FuncOp">]> {
  let summary = "AST-level return from function";
  let description = [{
    Returns a value from a function.

    Example:
    ```mlir
    polang_ast.return %0 : !polang.integer<64, signed>
    ```
  }];

  let arguments = (ins PolangAST_AnyTypeOrVar:$value);

  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// Variable reference operations
//===----------------------------------------------------------------------===//

def PolangAST_VarRefOp : PolangAST_Op<"var_ref", [Pure]> {
  let summary = "Unresolved variable reference";
  let description = [{
    References a variable by name. This is an unresolved reference that will
    be resolved by a later pass.

    Example:
    ```mlir
    %x = polang_ast.var_ref "x" : !polang_ast.typevar<1, any>
    ```
  }];

  let arguments = (ins StrAttr:$var_name);
  let results = (outs PolangAST_AnyTypeOrVar:$result);

  let assemblyFormat = "$var_name attr-dict `:` type($result)";
}

def PolangAST_QualifiedRefOp : PolangAST_Op<"qualified_ref", [Pure]> {
  let summary = "Unresolved qualified reference";
  let description = [{
    References a qualified name like module.member. This is an unresolved
    reference that will be resolved by a later pass.

    Example:
    ```mlir
    %pi = polang_ast.qualified_ref ["Math", "pi"] : !polang.float<64>
    ```
  }];

  let arguments = (ins StrArrayAttr:$path);
  let results = (outs PolangAST_AnyTypeOrVar:$result);

  let assemblyFormat = "$path attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Let expression operations
//===----------------------------------------------------------------------===//

def PolangAST_LetExprOp : PolangAST_Op<"let_expr", [RecursiveMemoryEffects]> {
  let summary = "Let-in expression with separate binding regions";
  let description = [{
    A let expression with separate regions for each binding and a body.
    Each binding region computes a value independently and yields it.
    The body region receives bound values as block arguments.

    Note: The body region is declared first in the region list (as MLIR requires
    the variadic region to be last), but in the assembly format we print
    bindings first followed by body for natural reading order.

    Example:
    ```mlir
    %result = polang_ast.let_expr ["x", "y"] -> !polang.integer<64, signed>
      binding {
        polang_ast.yield.binding %val1 : !polang.integer<64, signed>
      }
      binding {
        polang_ast.yield.binding %val2 : !polang.integer<64, signed>
      }
      body {
      ^bb0(%x: !polang.integer<64, signed>, %y: !polang.integer<64, signed>):
        %sum = polang_ast.add %x, %y : ...
        polang_ast.yield %sum : !polang.integer<64, signed>
      }
    ```
  }];

  let arguments = (ins StrArrayAttr:$var_names);
  let results = (outs PolangAST_AnyTypeOrVar:$result);
  // Body must come first since MLIR requires variadic regions to be last
  let regions = (region
    SizedRegion<1>:$body,
    VariadicRegion<SizedRegion<1>>:$bindings
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Module operations
//===----------------------------------------------------------------------===//

def PolangAST_ModuleOp : PolangAST_Op<"module", [
    Symbol,
    SymbolTable,
    IsolatedFromAbove,
    NoTerminator
  ]> {
  let summary = "Module declaration";
  let description = [{
    Declares a module containing functions and other definitions.

    Example:
    ```mlir
    polang_ast.module @Math {
      polang_ast.func @add(...) -> ... { ... }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def PolangAST_ImportOp : PolangAST_Op<"import", []> {
  let summary = "Import statement";
  let description = [{
    Imports a module, optionally with an alias.

    Example:
    ```mlir
    polang_ast.import @Math
    polang_ast.import @Math as @M
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$module_name, OptionalAttr<FlatSymbolRefAttr>:$alias);

  let hasCustomAssemblyFormat = 1;
}

#endif // POLANG_AST_OPS_TD
