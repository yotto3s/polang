//===- PolangOps.cpp - Polang operation implementation ----------*- C++ -*-===//
//
// This file implements the operations for the Polang dialect.
//
//===----------------------------------------------------------------------===//

#include "polang/Dialect/PolangOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "polang/Dialect/PolangDialect.h"
#include "polang/Dialect/PolangTypes.h"

using namespace mlir;
using namespace polang;

namespace {
/// Check if two types are compatible for verification purposes.
/// Types are compatible if they are equal OR if either is a type variable.
/// Type variables will be resolved by the type inference pass.
bool typesAreCompatible(Type t1, Type t2) {
  if (t1 == t2)
    return true;
  if (isa<TypeVarType>(t1) || isa<TypeVarType>(t2))
    return true;
  return false;
}
} // namespace

#include "polang/Dialect/PolangEnums.cpp.inc"

#define GET_OP_CLASSES
#include "polang/Dialect/PolangOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder& builder, OperationState& state, StringRef name,
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

ParseResult FuncOp::parse(OpAsmParser& parser, OperationState& result) {
  auto buildFuncType = [](Builder& builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag, std::string&) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter& p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(OpBuilder& builder, OperationState& state, StringRef callee,
                   TypeRange results, ValueRange operands) {
  state.addOperands(operands);
  state.addAttribute("callee",
                     SymbolRefAttr::get(builder.getContext(), callee));
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

void IfOp::build(OpBuilder& builder, OperationState& state, Type resultType,
                 Value condition) {
  state.addOperands(condition);
  state.addTypes(resultType);

  // Create then region
  Region* thenRegion = state.addRegion();
  thenRegion->push_back(new Block());

  // Create else region
  Region* elseRegion = state.addRegion();
  elseRegion->push_back(new Block());
}

ParseResult IfOp::parse(OpAsmParser& parser, OperationState& result) {
  OpAsmParser::UnresolvedOperand condition;
  Type resultType;

  if (parser.parseOperand(condition) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  if (parser.resolveOperand(condition, BoolType::get(parser.getContext()),
                            result.operands))
    return failure();

  result.addTypes(resultType);

  // Parse then region
  Region* thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // Parse else region
  if (parser.parseKeyword("else"))
    return failure();

  Region* elseRegion = result.addRegion();
  if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

void IfOp::print(OpAsmPrinter& p) {
  p << " " << getCondition() << " -> " << getResult().getType() << " ";
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p << " else ";
  p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

LogicalResult IfOp::verify() {
  // Check that both regions have terminators
  if (getThenRegion().empty() || getThenRegion().front().empty())
    return emitOpError("then region must not be empty");
  if (getElseRegion().empty() || getElseRegion().front().empty())
    return emitOpError("else region must not be empty");

  auto* thenTerminator = getThenRegion().front().getTerminator();
  auto* elseTerminator = getElseRegion().front().getTerminator();

  auto thenYield = dyn_cast<YieldOp>(thenTerminator);
  auto elseYield = dyn_cast<YieldOp>(elseTerminator);

  if (!thenYield)
    return emitOpError("then region must end with polang.yield");
  if (!elseYield)
    return emitOpError("else region must end with polang.yield");

  // Check that yield types match result type (allow type variables)
  if (!typesAreCompatible(thenYield.getValue().getType(),
                          getResult().getType()))
    return emitOpError("then branch yields ")
           << thenYield.getValue().getType() << " but if expects "
           << getResult().getType();

  if (!typesAreCompatible(elseYield.getValue().getType(),
                          getResult().getType()))
    return emitOpError("else branch yields ")
           << elseYield.getValue().getType() << " but if expects "
           << getResult().getType();

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp verifier
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto funcOp = dyn_cast<FuncOp>((*this)->getParentOp());
  if (!funcOp)
    return emitOpError("must be inside a polang.func");

  auto resultTypes = funcOp.getResultTypes();

  if (getValue()) {
    if (resultTypes.empty())
      return emitOpError("returns a value but function has no return type");
    // Allow type variables - they will be resolved by type inference pass
    if (!typesAreCompatible(getValue().getType(), resultTypes[0]))
      return emitOpError("returns ")
             << getValue().getType() << " but function expects "
             << resultTypes[0];
  } else {
    if (!resultTypes.empty())
      return emitOpError("must return a value of type ") << resultTypes[0];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp verifier
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  // Check if the reference comes from an alloca
  if (auto allocaOp = getRef().getDefiningOp<AllocaOp>()) {
    if (!allocaOp.getIsMutable())
      return emitOpError("cannot store to immutable variable '")
             << allocaOp.getName() << "'";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp verifier and SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verify() {
  // Basic structural checks - symbol verification is done by verifySymbolUses
  return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  auto funcOp =
      symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, getCalleeAttr());
  if (!funcOp)
    return emitOpError("references undefined function '") << getCallee() << "'";

  auto funcType = funcOp.getFunctionType();

  // Check argument count
  if (getOperands().size() != funcType.getNumInputs())
    return emitOpError("function '")
           << getCallee() << "' expects " << funcType.getNumInputs()
           << " argument(s) but got " << getOperands().size();

  // Check argument types (allow type variables)
  for (unsigned i = 0; i < getOperands().size(); ++i) {
    if (!typesAreCompatible(getOperands()[i].getType(), funcType.getInput(i)))
      return emitOpError("argument ")
             << (i + 1) << " has type " << getOperands()[i].getType()
             << " but function expects " << funcType.getInput(i);
  }

  // Check result type (allow type variables)
  if (!funcType.getResults().empty() && getNumResults() > 0) {
    if (!typesAreCompatible(getResult().getType(), funcType.getResult(0)))
      return emitOpError("result type ")
             << getResult().getType() << " does not match function return type "
             << funcType.getResult(0);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Arithmetic operation verifiers
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType()))
    return emitOpError("operand types must be compatible");
  if (!typesAreCompatible(getLhs().getType(), getResult().getType()))
    return emitOpError("result type must be compatible with operands");
  return success();
}

LogicalResult SubOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType()))
    return emitOpError("operand types must be compatible");
  if (!typesAreCompatible(getLhs().getType(), getResult().getType()))
    return emitOpError("result type must be compatible with operands");
  return success();
}

LogicalResult MulOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType()))
    return emitOpError("operand types must be compatible");
  if (!typesAreCompatible(getLhs().getType(), getResult().getType()))
    return emitOpError("result type must be compatible with operands");
  return success();
}

LogicalResult DivOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType()))
    return emitOpError("operand types must be compatible");
  if (!typesAreCompatible(getLhs().getType(), getResult().getType()))
    return emitOpError("result type must be compatible with operands");
  return success();
}

//===----------------------------------------------------------------------===//
// Comparison operation verifier
//===----------------------------------------------------------------------===//

LogicalResult CmpOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType()))
    return emitOpError("comparison operand types must be compatible");
  return success();
}

//===----------------------------------------------------------------------===//
// InferTypeOpInterface implementations
//===----------------------------------------------------------------------===//

// Note: Arithmetic ops need explicit result type specification in MLIRGen
// since we removed SameOperandsAndResultType to allow type variables.
// CmpOp always returns bool, so it infers the result type automatically.

// Note: LoadOp does not implement InferTypeOpInterface because the memref
// element type is the LLVM type (i64), not the Polang type (!polang.int).
// The Polang type is stored in AllocaOp's elementType attribute, which requires
// walking to the defining op. MLIRGen already specifies the type explicitly.
