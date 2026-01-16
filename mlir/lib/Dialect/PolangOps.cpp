//===- PolangOps.cpp - Polang operation implementation ----------*- C++ -*-===//
//
// This file implements the operations for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#include "polang/Dialect/PolangOps.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace polang;

#include "polang/Dialect/PolangEnums.cpp.inc"

#define GET_OP_CLASSES
#include "polang/Dialect/PolangOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<StringRef> captures) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));

  if (!captures.empty()) {
    SmallVector<Attribute> captureAttrs;
    for (StringRef capture : captures) {
      captureAttrs.push_back(builder.getStringAttr(capture));
    }
    state.addAttribute(getCapturesAttrName(state.name),
                       builder.getArrayAttr(captureAttrs));
  }

  state.addRegion();
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(OpBuilder &builder, OperationState &state, StringRef callee,
                   TypeRange results, ValueRange operands) {
  state.addOperands(operands);
  state.addAttribute("callee", SymbolRefAttr::get(builder.getContext(), callee));
  state.addTypes(results);
}

CallInterfaceCallable CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", llvm::cast<SymbolRefAttr>(callee));
}

Operation::operand_range CallOp::getArgOperands() { return getOperands(); }

MutableOperandRange CallOp::getArgOperandsMutable() {
  return getOperandsMutable();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                 Value condition) {
  state.addOperands(condition);
  state.addTypes(resultType);

  // Create then region
  Region *thenRegion = state.addRegion();
  thenRegion->push_back(new Block());

  // Create else region
  Region *elseRegion = state.addRegion();
  elseRegion->push_back(new Block());
}

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand condition;
  Type resultType;

  if (parser.parseOperand(condition) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  if (parser.resolveOperand(condition,
                            BoolType::get(parser.getContext()), result.operands))
    return failure();

  result.addTypes(resultType);

  // Parse then region
  Region *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // Parse else region
  if (parser.parseKeyword("else"))
    return failure();

  Region *elseRegion = result.addRegion();
  if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition() << " -> " << getResult().getType() << " ";
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p << " else ";
  p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}
