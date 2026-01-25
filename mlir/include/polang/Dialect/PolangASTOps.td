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
// Operations will be added in subsequent steps
//===----------------------------------------------------------------------===//

#endif // POLANG_AST_OPS_TD
