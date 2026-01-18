//===- PolangOps.cpp - Polang operation implementation ----------*- C++ -*-===//
//
// This file implements the operations for the Polang dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers and generated code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

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
  if (t1 == t2) {
    return true;
  }
  if (isa<TypeVarType>(t1) || isa<TypeVarType>(t2)) {
    return true;
  }
  return false;
}
} // namespace

#include "polang/Dialect/PolangEnums.cpp.inc"

#define GET_OP_CLASSES
#include "polang/Dialect/PolangOps.cpp.inc"

#pragma GCC diagnostic pop

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
  (void)builder; // Unused, but required by MLIR interface
  state.addOperands(condition);
  state.addTypes(resultType);

  // Create then region
  Region* thenRegion = state.addRegion();
  thenRegion->push_back(new Block());

  // Create else region
  Region* elseRegion = state.addRegion();
  elseRegion->push_back(new Block());
}

LogicalResult IfOp::verify() {
  // Check that both regions have terminators
  if (getThenRegion().empty() || getThenRegion().front().empty()) {
    return emitOpError("then region must not be empty");
  }
  if (getElseRegion().empty() || getElseRegion().front().empty()) {
    return emitOpError("else region must not be empty");
  }

  auto* thenTerminator = getThenRegion().front().getTerminator();
  auto* elseTerminator = getElseRegion().front().getTerminator();

  auto thenYield = dyn_cast<YieldOp>(thenTerminator);
  auto elseYield = dyn_cast<YieldOp>(elseTerminator);

  if (!thenYield) {
    return emitOpError("then region must end with polang.yield");
  }
  if (!elseYield) {
    return emitOpError("else region must end with polang.yield");
  }

  // Check that yield types match result type (allow type variables)
  if (!typesAreCompatible(thenYield.getValue().getType(),
                          getResult().getType())) {
    return emitOpError("then branch yields ")
           << thenYield.getValue().getType() << " but if expects "
           << getResult().getType();
  }

  if (!typesAreCompatible(elseYield.getValue().getType(),
                          getResult().getType())) {
    return emitOpError("else branch yields ")
           << elseYield.getValue().getType() << " but if expects "
           << getResult().getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp verifier
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto funcOp = dyn_cast<FuncOp>((*this)->getParentOp());
  if (!funcOp) {
    return emitOpError("must be inside a polang.func");
  }

  auto resultTypes = funcOp.getResultTypes();

  if (getValue()) {
    if (resultTypes.empty()) {
      return emitOpError("returns a value but function has no return type");
    }
    // Allow type variables - they will be resolved by type inference pass
    if (!typesAreCompatible(getValue().getType(), resultTypes[0])) {
      return emitOpError("returns ")
             << getValue().getType() << " but function expects "
             << resultTypes[0];
    }
  } else {
    if (!resultTypes.empty()) {
      return emitOpError("must return a value of type ") << resultTypes[0];
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp verifier
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  // Check if the reference comes from an alloca
  if (auto allocaOp = getRef().getDefiningOp<AllocaOp>()) {
    if (!allocaOp.getIsMutable()) {
      return emitOpError("cannot store to immutable variable '")
             << allocaOp.getName() << "'";
    }
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
  if (!funcOp) {
    return emitOpError("references undefined function '") << getCallee() << "'";
  }

  auto funcType = funcOp.getFunctionType();

  // Check argument count
  if (getOperands().size() != funcType.getNumInputs()) {
    return emitOpError("function '")
           << getCallee() << "' expects " << funcType.getNumInputs()
           << " argument(s) but got " << getOperands().size();
  }

  // Check argument types (allow type variables)
  for (unsigned i = 0; i < getOperands().size(); ++i) {
    if (!typesAreCompatible(getOperands()[i].getType(), funcType.getInput(i))) {
      return emitOpError("argument ")
             << (i + 1) << " has type " << getOperands()[i].getType()
             << " but function expects " << funcType.getInput(i);
    }
  }

  // Check result type (allow type variables)
  if (!funcType.getResults().empty() && getNumResults() > 0) {
    if (!typesAreCompatible(getResult().getType(), funcType.getResult(0))) {
      return emitOpError("result type ")
             << getResult().getType() << " does not match function return type "
             << funcType.getResult(0);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Arithmetic operation verifiers
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType())) {
    return emitOpError("operand types must be compatible");
  }
  if (!typesAreCompatible(getLhs().getType(), getResult().getType())) {
    return emitOpError("result type must be compatible with operands");
  }
  return success();
}

LogicalResult SubOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType())) {
    return emitOpError("operand types must be compatible");
  }
  if (!typesAreCompatible(getLhs().getType(), getResult().getType())) {
    return emitOpError("result type must be compatible with operands");
  }
  return success();
}

LogicalResult MulOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType())) {
    return emitOpError("operand types must be compatible");
  }
  if (!typesAreCompatible(getLhs().getType(), getResult().getType())) {
    return emitOpError("result type must be compatible with operands");
  }
  return success();
}

LogicalResult DivOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType())) {
    return emitOpError("operand types must be compatible");
  }
  if (!typesAreCompatible(getLhs().getType(), getResult().getType())) {
    return emitOpError("result type must be compatible with operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp verifier
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  // Allow type variables - they will be resolved by type inference pass
  if (isa<TypeVarType>(inputType) || isa<TypeVarType>(resultType)) {
    return success();
  }

  // Both types must be numeric (not bool)
  const bool inputIsNumeric =
      isa<polang::IntegerType>(inputType) || isa<polang::FloatType>(inputType);
  const bool resultIsNumeric = isa<polang::IntegerType>(resultType) ||
                               isa<polang::FloatType>(resultType);

  if (!inputIsNumeric) {
    return emitOpError("input type must be numeric, got ") << inputType;
  }
  if (!resultIsNumeric) {
    return emitOpError("result type must be numeric, got ") << resultType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Comparison operation verifier
//===----------------------------------------------------------------------===//

LogicalResult CmpOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType())) {
    return emitOpError("comparison operand types must be compatible");
  }
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

//===----------------------------------------------------------------------===//
// ConstantIntegerOp custom print/parse
//===----------------------------------------------------------------------===//

void ConstantIntegerOp::print(OpAsmPrinter& p) {
  // Print just the integer value (without the IntegerAttr's type)
  p << " " << getValueAttr().getValue();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << " : ";
  p.printType(getResult().getType());
}

ParseResult ConstantIntegerOp::parse(OpAsmParser& parser,
                                     OperationState& result) {
  APInt value;
  Type resultType;

  // Parse the raw integer value
  if (parser.parseInteger(value) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType)) {
    return failure();
  }

  // Create the IntegerAttr with the appropriate bit width
  unsigned width = 64; // Default width for type variables
  if (auto intType = dyn_cast<polang::IntegerType>(resultType)) {
    width = intType.getWidth();
  } else if (!isa<TypeVarType>(resultType)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected polang.integer or typevar type");
  }
  auto attr = IntegerAttr::get(
      mlir::IntegerType::get(parser.getContext(), width), value);
  result.addAttribute("value", attr);
  result.addTypes(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantFloatOp custom print/parse
//===----------------------------------------------------------------------===//

void ConstantFloatOp::print(OpAsmPrinter& p) {
  // Print the float value
  p << " ";
  p.printFloat(getValueAttr().getValue());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << " : ";
  p.printType(getResult().getType());
}

ParseResult ConstantFloatOp::parse(OpAsmParser& parser,
                                   OperationState& result) {
  double value;
  Type resultType;

  // Parse the raw float value
  if (parser.parseFloat(value) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType)) {
    return failure();
  }

  // Create the FloatAttr with the appropriate type
  mlir::Type attrType =
      Float64Type::get(parser.getContext()); // Default for type vars
  if (auto floatType = dyn_cast<polang::FloatType>(resultType)) {
    if (floatType.getWidth() == 32) {
      attrType = Float32Type::get(parser.getContext());
    } else {
      attrType = Float64Type::get(parser.getContext());
    }
  } else if (!isa<TypeVarType>(resultType)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected polang.float or typevar type");
  }
  auto attr = FloatAttr::get(attrType, value);
  result.addAttribute("value", attr);
  result.addTypes(resultType);
  return success();
}
