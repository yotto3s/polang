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

#endif // POLANG_AST_OPS_TD
