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
/// Types are compatible if they are equal OR if either is a type parameter.
/// Type parameters will be resolved by the monomorphization pass.
bool typesAreCompatible(Type t1, Type t2) {
  if (t1 == t2) {
    return true;
  }
  if (isa<TypeParamType>(t1) || isa<TypeParamType>(t2)) {
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
// GenericFuncOp
//===----------------------------------------------------------------------===//

void GenericFuncOp::build(OpBuilder& builder, OperationState& state,
                          StringRef name, FunctionType type,
                          ArrayRef<StringRef> typeParams,
                          ArrayRef<StringRef> typeParamBounds,
                          ArrayRef<StringRef> captures) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));

  SmallVector<Attribute> paramAttrs;
  for (StringRef param : typeParams) {
    paramAttrs.push_back(builder.getStringAttr(param));
  }
  state.addAttribute(getTypeParamsAttrName(state.name),
                     builder.getArrayAttr(paramAttrs));

  if (!typeParamBounds.empty()) {
    SmallVector<Attribute> boundAttrs;
    for (StringRef bound : typeParamBounds) {
      boundAttrs.push_back(builder.getStringAttr(bound));
    }
    state.addAttribute(getTypeParamBoundsAttrName(state.name),
                       builder.getArrayAttr(boundAttrs));
  }

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

/// Parse type parameters: <a, b: Numeric>
/// Type param names are bare identifiers in the textual MLIR format.
/// They correspond to ML-style 'a, 'b names stored internally.
static ParseResult
parseTypeParams(OpAsmParser& parser,
                SmallVectorImpl<std::string>& typeParamNames,
                SmallVectorImpl<std::string>& typeParamBounds) {
  if (parser.parseLess()) {
    return failure();
  }

  auto parseOneParam = [&]() -> ParseResult {
    StringRef nameRef;
    if (parser.parseKeyword(&nameRef)) {
      return failure();
    }
    typeParamNames.push_back(nameRef.str());

    // Parse optional bound: ": Numeric"
    std::string bound;
    if (succeeded(parser.parseOptionalColon())) {
      StringRef boundRef;
      if (parser.parseKeyword(&boundRef)) {
        return failure();
      }
      bound = boundRef.str();
    }
    typeParamBounds.push_back(bound);
    return success();
  };

  if (parseOneParam()) {
    return failure();
  }
  while (succeeded(parser.parseOptionalComma())) {
    if (parseOneParam()) {
      return failure();
    }
  }

  return parser.parseGreater();
}

ParseResult GenericFuncOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse the function name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  }

  // Parse type parameters <'a, 'b: Numeric>
  SmallVector<std::string> typeParamNames;
  SmallVector<std::string> typeParamBounds;
  if (parseTypeParams(parser, typeParamNames, typeParamBounds)) {
    return failure();
  }

  auto& builder = parser.getBuilder();
  SmallVector<Attribute> paramAttrs;
  for (const auto& name : typeParamNames) {
    paramAttrs.push_back(builder.getStringAttr(name));
  }
  result.addAttribute(getTypeParamsAttrName(result.name),
                      builder.getArrayAttr(paramAttrs));

  // Only add bounds if at least one is non-empty
  bool hasBounds = llvm::any_of(
      typeParamBounds, [](const std::string& b) { return !b.empty(); });
  if (hasBounds) {
    SmallVector<Attribute> boundAttrs;
    for (const auto& bound : typeParamBounds) {
      boundAttrs.push_back(builder.getStringAttr(bound));
    }
    result.addAttribute(getTypeParamBoundsAttrName(result.name),
                        builder.getArrayAttr(boundAttrs));
  }

  // Parse the function signature (args, return type)
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  bool isVariadic = false;

  if (function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, args, isVariadic, resultTypes,
          resultAttrs)) {
    return failure();
  }

  // Build function type from parsed args and results
  SmallVector<Type> argTypes;
  for (const auto& arg : args) {
    argTypes.push_back(arg.type);
  }
  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(funcType));

  // Parse optional attributes
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return failure();
  }

  // Add arg/result attributes
  SmallVector<Attribute> argAttrs;
  for (const auto& arg : args) {
    auto namedAttrList = arg.attrs.getValue();
    argAttrs.push_back(
        namedAttrList.empty()
            ? DictionaryAttr()
            : DictionaryAttr::get(parser.getContext(), namedAttrList));
  }
  if (!llvm::all_of(argAttrs, [](Attribute a) { return !a; })) {
    result.addAttribute(getArgAttrsAttrName(result.name),
                        builder.getArrayAttr(argAttrs));
  }
  if (!resultAttrs.empty() &&
      !llvm::all_of(resultAttrs, [](DictionaryAttr a) { return !a; })) {
    SmallVector<Attribute> resAttrVec(resultAttrs.begin(), resultAttrs.end());
    result.addAttribute(getResAttrsAttrName(result.name),
                        builder.getArrayAttr(resAttrVec));
  }

  // Parse body
  auto* body = result.addRegion();
  if (parser.parseRegion(*body, args)) {
    return failure();
  }
  if (body->empty()) {
    body->push_back(new Block());
  }

  return success();
}

void GenericFuncOp::print(OpAsmPrinter& p) {
  // Print function name
  p << " ";
  p.printSymbolName(getSymName());

  // Print type parameters <a, b: Numeric>
  auto typeParams = getTypeParams();
  auto typeParamBounds = getTypeParamBounds();
  p << "<";
  for (size_t i = 0; i < typeParams.size(); ++i) {
    if (i > 0) {
      p << ", ";
    }
    p << cast<StringAttr>(typeParams[i]).getValue();
    if (typeParamBounds) {
      auto bound = cast<StringAttr>((*typeParamBounds)[i]).getValue();
      if (!bound.empty()) {
        p << ": " << bound;
      }
    }
  }
  p << ">";

  // Print function signature (args, return type, body)
  auto funcType = getFunctionType();
  function_interface_impl::printFunctionSignature(
      p, *this, funcType.getInputs(), /*isVariadic=*/false,
      funcType.getResults());

  // Print attributes (excluding known ones)
  function_interface_impl::printFunctionAttributes(
      p, *this,
      {getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName(),
       getTypeParamsAttrName(), getTypeParamBoundsAttrName(),
       getCapturesAttrName()});

  // Print body
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// InstantiateOp
//===----------------------------------------------------------------------===//

void InstantiateOp::build(OpBuilder& builder, OperationState& state,
                          StringRef callee, TypeRange results,
                          ValueRange operands,
                          ArrayRef<StringRef> typeParamNames,
                          ArrayRef<Type> typeBindings) {
  state.addOperands(operands);
  state.addAttribute("callee",
                     SymbolRefAttr::get(builder.getContext(), callee));
  state.addTypes(results);

  SmallVector<Attribute> nameAttrs;
  for (StringRef name : typeParamNames) {
    nameAttrs.push_back(builder.getStringAttr(name));
  }
  state.addAttribute(getTypeParamNamesAttrName(state.name),
                     builder.getArrayAttr(nameAttrs));

  SmallVector<Attribute> typeAttrs;
  for (Type t : typeBindings) {
    typeAttrs.push_back(TypeAttr::get(t));
  }
  state.addAttribute(getTypeBindingsAttrName(state.name),
                     builder.getArrayAttr(typeAttrs));
}

CallInterfaceCallable InstantiateOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void InstantiateOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", llvm::cast<SymbolRefAttr>(callee));
}

Operation::operand_range InstantiateOp::getArgOperands() {
  return getOperands();
}

MutableOperandRange InstantiateOp::getArgOperandsMutable() {
  return getOperandsMutable();
}

FunctionType InstantiateOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

/// Parse type bindings: <a = !polang.integer<64, signed>, b = !polang.bool>
/// Type param names are bare identifiers in the textual MLIR format.
static ParseResult
parseTypeBindings(OpAsmParser& parser,
                  SmallVectorImpl<std::string>& typeParamNames,
                  SmallVectorImpl<Type>& typeBindings) {
  if (parser.parseLess()) {
    return failure();
  }

  auto parseOneBinding = [&]() -> ParseResult {
    StringRef nameRef;
    if (parser.parseKeyword(&nameRef)) {
      return failure();
    }
    typeParamNames.push_back(nameRef.str());

    // Parse = type
    if (parser.parseEqual()) {
      return failure();
    }
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    typeBindings.push_back(type);
    return success();
  };

  if (parseOneBinding()) {
    return failure();
  }
  while (succeeded(parser.parseOptionalComma())) {
    if (parseOneBinding()) {
      return failure();
    }
  }

  return parser.parseGreater();
}

ParseResult InstantiateOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse callee
  FlatSymbolRefAttr calleeAttr;
  if (parser.parseAttribute(calleeAttr, "callee", result.attributes)) {
    return failure();
  }

  // Parse type bindings <'a = !polang.integer<64, signed>>
  SmallVector<std::string> typeParamNames;
  SmallVector<Type> typeBindings;
  if (parseTypeBindings(parser, typeParamNames, typeBindings)) {
    return failure();
  }

  auto& builder = parser.getBuilder();
  SmallVector<Attribute> nameAttrs;
  for (const auto& name : typeParamNames) {
    nameAttrs.push_back(builder.getStringAttr(name));
  }
  result.addAttribute(InstantiateOp::getTypeParamNamesAttrName(result.name),
                      builder.getArrayAttr(nameAttrs));

  SmallVector<Attribute> typeAttrs;
  for (Type t : typeBindings) {
    typeAttrs.push_back(TypeAttr::get(t));
  }
  result.addAttribute(InstantiateOp::getTypeBindingsAttrName(result.name),
                      builder.getArrayAttr(typeAttrs));

  // Parse operands
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseLParen()) {
    return failure();
  }
  if (parser.parseOptionalRParen()) {
    if (parser.parseOperandList(operands) || parser.parseRParen()) {
      return failure();
    }
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Parse functional type : (operand types) -> result types
  FunctionType funcType;
  if (parser.parseColonType(funcType)) {
    return failure();
  }

  // Resolve operands
  if (parser.resolveOperands(operands, funcType.getInputs(),
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  result.addTypes(funcType.getResults());
  return success();
}

void InstantiateOp::print(OpAsmPrinter& p) {
  p << " @" << getCallee();

  // Print type bindings <a = !polang.integer<64, signed>>
  auto names = getTypeParamNames();
  auto bindings = getTypeBindings();
  p << "<";
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) {
      p << ", ";
    }
    p << cast<StringAttr>(names[i]).getValue() << " = ";
    p.printType(cast<TypeAttr>(bindings[i]).getValue());
  }
  p << ">";

  // Print operands
  p << "(";
  p.printOperands(getOperands());
  p << ")";

  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"callee", "type_param_names", "type_bindings"});
  p << " : ";
  p.printFunctionalType(getOperandTypes(), getResultTypes());
}

LogicalResult InstantiateOp::verify() {
  // Check that type_param_names and type_bindings have the same size
  if (getTypeParamNames().size() != getTypeBindings().size()) {
    return emitOpError("type_param_names count (")
           << getTypeParamNames().size() << ") must match type_bindings count ("
           << getTypeBindings().size() << ")";
  }
  return success();
}

LogicalResult
InstantiateOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  auto genericFuncOp = symbolTable.lookupNearestSymbolFrom<GenericFuncOp>(
      *this, getCalleeAttr());
  if (!genericFuncOp) {
    return emitOpError("references undefined generic function '")
           << getCallee() << "'";
  }

  auto funcType = genericFuncOp.getFunctionType();

  // Check argument count
  if (getOperands().size() != funcType.getNumInputs()) {
    return emitOpError("generic function '")
           << getCallee() << "' expects " << funcType.getNumInputs()
           << " argument(s) but got " << getOperands().size();
  }

  return success();
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
  auto* parentOp = (*this)->getParentOp();
  ArrayRef<Type> resultTypes;

  if (auto funcOp = dyn_cast<FuncOp>(parentOp)) {
    resultTypes = funcOp.getResultTypes();
  } else if (auto genericFuncOp = dyn_cast<GenericFuncOp>(parentOp)) {
    resultTypes = genericFuncOp.getResultTypes();
  } else {
    return emitOpError("must be inside a polang.func or polang.generic_func");
  }

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
// CallOp verifier and SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verify() {
  // Basic structural checks - symbol verification is done by verifySymbolUses
  return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  FunctionType funcType;
  auto funcOp =
      symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, getCalleeAttr());
  if (funcOp) {
    funcType = funcOp.getFunctionType();
  } else {
    // Also check for GenericFuncOp
    auto genericFuncOp = symbolTable.lookupNearestSymbolFrom<GenericFuncOp>(
        *this, getCalleeAttr());
    if (!genericFuncOp) {
      return emitOpError("references undefined function '")
             << getCallee() << "'";
    }
    funcType = genericFuncOp.getFunctionType();
  }

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

  // Allow type parameters - they will be resolved by monomorphization
  if (isa<TypeParamType>(inputType) || isa<TypeParamType>(resultType)) {
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
// MLIRGen already specifies the type explicitly.

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
  unsigned width = 64; // Default width for type parameters
  if (auto intType = dyn_cast<polang::IntegerType>(resultType)) {
    width = intType.getWidth();
  } else if (!isa<TypeParamType>(resultType)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected polang.integer or type_param type");
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
  } else if (!isa<TypeParamType>(resultType)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected polang.float or type_param type");
  }
  auto attr = FloatAttr::get(attrType, value);
  result.addAttribute("value", attr);
  result.addTypes(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// LetExprOp
//===----------------------------------------------------------------------===//

void LetExprOp::print(OpAsmPrinter& p) {
  p << " ";
  p.printAttribute(getVarNamesAttr());
  p << " -> " << getResult().getType();

  for (Region& binding : getBindings()) {
    p << "\n  binding ";
    p.printRegion(binding, /*printEntryBlockArgs=*/false);
  }

  p << "\n  body ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"var_names"});
}

ParseResult LetExprOp::parse(OpAsmParser& parser, OperationState& result) {
  ArrayAttr varNamesAttr;
  Type resultType;

  if (parser.parseAttribute(varNamesAttr, "var_names", result.attributes) ||
      parser.parseArrow() || parser.parseType(resultType)) {
    return failure();
  }
  result.addTypes(resultType);

  size_t numBindings = varNamesAttr.size();
  SmallVector<std::unique_ptr<Region>> bindingRegions;
  for (size_t i = 0; i < numBindings; ++i) {
    if (parser.parseKeyword("binding")) {
      return failure();
    }
    auto bindingRegion = std::make_unique<Region>();
    if (parser.parseRegion(*bindingRegion, /*arguments=*/{})) {
      return failure();
    }
    if (bindingRegion->empty()) {
      bindingRegion->push_back(new Block());
    }
    bindingRegions.push_back(std::move(bindingRegion));
  }

  if (parser.parseKeyword("body")) {
    return failure();
  }
  Region* body = result.addRegion();
  if (parser.parseRegion(*body)) {
    return failure();
  }
  if (body->empty()) {
    body->push_back(new Block());
  }

  for (auto& bindingRegion : bindingRegions) {
    Region* binding = result.addRegion();
    binding->takeBody(*bindingRegion);
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

LogicalResult LetExprOp::verify() {
  if (getBindings().empty()) {
    return emitOpError("must have at least one binding region");
  }

  if (getBindings().size() != getVarNames().size()) {
    return emitOpError("number of binding regions (")
           << getBindings().size() << ") must match var_names count ("
           << getVarNames().size() << ")";
  }

  SmallVector<Type> bindingTypes;
  for (auto [idx, binding] : llvm::enumerate(getBindings())) {
    if (binding.empty() || binding.front().empty()) {
      return emitOpError("binding region #") << idx << " must not be empty";
    }

    auto* terminator = binding.front().getTerminator();
    auto yieldBinding = dyn_cast<YieldBindingOp>(terminator);
    if (!yieldBinding) {
      return emitOpError("binding region #")
             << idx << " must end with polang.yield.binding";
    }
    bindingTypes.push_back(yieldBinding.getValue().getType());
  }

  if (getBody().empty() || getBody().front().empty()) {
    return emitOpError("body region must not be empty");
  }

  Block& bodyEntry = getBody().front();
  if (bodyEntry.getNumArguments() != bindingTypes.size()) {
    return emitOpError("body block argument count (")
           << bodyEntry.getNumArguments() << ") doesn't match binding count ("
           << bindingTypes.size() << ")";
  }
  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(bodyEntry.getArguments(), bindingTypes))) {
    auto [blockArg, yieldType] = pair;
    if (!typesAreCompatible(blockArg.getType(), yieldType)) {
      return emitOpError("body block argument #")
             << idx << " type " << blockArg.getType()
             << " doesn't match binding yield type " << yieldType;
    }
  }

  auto* bodyTerminator = getBody().front().getTerminator();
  auto yieldOp = dyn_cast<YieldOp>(bodyTerminator);
  if (!yieldOp) {
    return emitOpError("body must end with polang.yield");
  }

  if (!typesAreCompatible(yieldOp.getValue().getType(),
                          getResult().getType())) {
    return emitOpError("yield type ")
           << yieldOp.getValue().getType() << " doesn't match result type "
           << getResult().getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

ParseResult GlobalOp::parse(OpAsmParser& parser, OperationState& result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  }

  // Parse optional {is_external}
  if (succeeded(parser.parseOptionalLBrace())) {
    if (parser.parseKeyword("is_external") || parser.parseRBrace()) {
      return failure();
    }
    result.addAttribute("is_external", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseColon()) {
    return failure();
  }

  TypeAttr typeAttr;
  if (parser.parseAttribute(typeAttr, "type", result.attributes)) {
    return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Parse optional initializer region
  auto* region = result.addRegion();
  auto parseResult = parser.parseOptionalRegion(*region);
  if (parseResult.has_value() && failed(*parseResult)) {
    return failure();
  }

  return success();
}

void GlobalOp::print(OpAsmPrinter& p) {
  p << ' ';
  p.printSymbolName(getSymName());
  if (getIsExternal()) {
    p << " {is_external}";
  }
  p << " : " << getType();
  SmallVector<StringRef> elidedAttrs = {SymbolTable::getSymbolAttrName(),
                                        "type", "is_external"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Print initializer region if present
  if (!getInitializer().empty()) {
    p << ' ';
    p.printRegion(getInitializer());
  }
}

LogicalResult GlobalOp::verify() {
  if (getIsExternal()) {
    // External globals must not have an initializer region
    if (!getInitializer().empty()) {
      return emitOpError("external global must not have an initializer region");
    }
    return success();
  }

  // Non-external globals with an initializer must have exactly one block
  if (!getInitializer().empty()) {
    auto& region = getInitializer();
    if (!region.hasOneBlock()) {
      return emitOpError("initializer region must have exactly one block");
    }

    // Block must end with YieldGlobalOp
    auto& block = region.front();
    if (block.empty()) {
      return emitOpError("initializer block must not be empty");
    }
    auto yieldOp = dyn_cast<YieldGlobalOp>(block.getTerminator());
    if (!yieldOp) {
      return emitOpError("initializer block must terminate with "
                         "'polang.yield.global'");
    }

    // Yielded type must match GlobalOp's type
    if (yieldOp.getValue().getType() != getType()) {
      return emitOpError("yielded type ")
             << yieldOp.getValue().getType() << " doesn't match global type "
             << getType();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalStoreOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalStoreOp::verify() { return success(); }

LogicalResult
GlobalStoreOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  auto globalOp =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getGlobalNameAttr());
  if (!globalOp) {
    return emitOpError("references undefined global '")
           << getGlobalName() << "'";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalLoadOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalLoadOp::verify() { return success(); }

LogicalResult
GlobalLoadOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  auto globalOp =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getGlobalNameAttr());
  if (!globalOp) {
    return emitOpError("references undefined global '")
           << getGlobalName() << "'";
  }
  return success();
}
