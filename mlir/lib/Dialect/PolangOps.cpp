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
/// Types are compatible if they are equal OR if either is a type variable
/// or type parameter. Type variables will be resolved by type inference,
/// and type parameters will be resolved by monomorphization.
bool typesAreCompatible(Type t1, Type t2) {
  if (t1 == t2) {
    return true;
  }
  if (llvm::isa<TypeVarType>(t1) || llvm::isa<TypeVarType>(t2)) {
    return true;
  }
  if (llvm::isa<TypeParamType>(t1) || llvm::isa<TypeParamType>(t2)) {
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

void CallOp::build(OpBuilder& builder, OperationState& state, StringRef callee,
                   ArrayAttr typeArgs, TypeRange results, ValueRange operands) {
  state.addOperands(operands);
  state.addAttribute("callee",
                     SymbolRefAttr::get(builder.getContext(), callee));
  if (typeArgs) {
    state.addAttribute("type_args", typeArgs);
  }
  state.addTypes(results);
}

void CallOp::print(OpAsmPrinter& p) {
  p << " @" << getCallee();

  // Print type arguments if present
  if (auto typeArgs = getTypeArgs()) {
    p << "<[";
    llvm::interleaveComma(*typeArgs, p, [&](Attribute attr) {
      p.printAttribute(attr);
    });
    p << "]>";
  }

  // Print operands
  p << "(";
  p.printOperands(getOperands());
  p << ")";

  // Print optional attributes (excluding callee and type_args)
  SmallVector<StringRef> elidedAttrs = {"callee"};
  if (getTypeArgs()) {
    elidedAttrs.push_back("type_args");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Print function type
  p << " : (";
  llvm::interleaveComma(getOperandTypes(), p, [&](Type type) {
    p.printType(type);
  });
  p << ") -> ";
  if (getNumResults() == 0) {
    p << "()";
  } else {
    p.printType(getResult().getType());
  }
}

ParseResult CallOp::parse(OpAsmParser& parser, OperationState& result) {
  FlatSymbolRefAttr calleeAttr;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  Type resultType;

  // Parse callee symbol
  if (parser.parseAttribute(calleeAttr, "callee", result.attributes)) {
    return failure();
  }

  // Parse optional type arguments: <[type1, type2, ...]>
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseLSquare()) {
      return failure();
    }

    SmallVector<Attribute> typeArgs;
    if (failed(parser.parseOptionalRSquare())) {
      do {
        Type type;
        if (parser.parseType(type)) {
          return failure();
        }
        typeArgs.push_back(TypeAttr::get(type));
      } while (succeeded(parser.parseOptionalComma()));

      if (parser.parseRSquare()) {
        return failure();
      }
    }

    if (parser.parseGreater()) {
      return failure();
    }

    if (!typeArgs.empty()) {
      result.addAttribute("type_args",
                          parser.getBuilder().getArrayAttr(typeArgs));
    }
  }

  // Parse operands
  if (parser.parseLParen()) {
    return failure();
  }

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(operands) || parser.parseRParen()) {
      return failure();
    }
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Parse function type: (operand_types) -> result_type
  if (parser.parseColon() || parser.parseLParen()) {
    return failure();
  }

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseTypeList(operandTypes) || parser.parseRParen()) {
      return failure();
    }
  }

  if (parser.parseArrow()) {
    return failure();
  }

  // Parse result type (could be () for void or a single type)
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseRParen()) {
      return failure();
    }
    // No result type
  } else {
    if (parser.parseType(resultType)) {
      return failure();
    }
    result.addTypes(resultType);
  }

  // Resolve operands
  if (parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             result.operands)) {
    return failure();
  }

  return success();
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
  llvm::ArrayRef<Type> resultTypes;

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
  auto typeArgs = getTypeArgs();

  // If type arguments are provided, must call a generic function
  if (typeArgs) {
    auto genericFuncOp =
        symbolTable.lookupNearestSymbolFrom<GenericFuncOp>(*this,
                                                          getCalleeAttr());
    if (!genericFuncOp) {
      // Check if it's a regular function (error: type args on non-generic)
      auto funcOp =
          symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, getCalleeAttr());
      if (funcOp) {
        return emitOpError("type arguments provided but '")
               << getCallee() << "' is not a generic function";
      }
      return emitOpError("references undefined function '")
             << getCallee() << "'";
    }

    // Check type argument count matches type parameter count
    unsigned numTypeParams = genericFuncOp.getNumTypeParams();
    if (typeArgs->size() != numTypeParams) {
      return emitOpError("expects ")
             << numTypeParams << " type argument(s) but got "
             << typeArgs->size();
    }

    // Check that each type argument satisfies its constraint
    for (size_t i = 0; i < typeArgs->size(); ++i) {
      auto typeArg = llvm::cast<TypeAttr>((*typeArgs)[i]).getValue();
      auto constraint = genericFuncOp.getConstraintAt(i);
      auto paramName =
          llvm::cast<StringAttr>(genericFuncOp.getTypeParams()[i]).getValue();

      // Check constraint satisfaction
      bool satisfies = false;
      switch (constraint) {
      case TypeParamKind::Any:
        satisfies = llvm::isa<polang::IntegerType>(typeArg) ||
                    llvm::isa<polang::FloatType>(typeArg) ||
                    llvm::isa<BoolType>(typeArg);
        break;
      case TypeParamKind::Numeric:
        satisfies = llvm::isa<polang::IntegerType>(typeArg) ||
                    llvm::isa<polang::FloatType>(typeArg);
        break;
      case TypeParamKind::Integer:
        satisfies = llvm::isa<polang::IntegerType>(typeArg);
        break;
      case TypeParamKind::Float:
        satisfies = llvm::isa<polang::FloatType>(typeArg);
        break;
      }

      if (!satisfies) {
        return emitOpError("type argument ")
               << typeArg << " does not satisfy '"
               << stringifyTypeParamKind(constraint)
               << "' constraint for type parameter '" << paramName << "'";
      }
    }

    // For generic calls, we skip argument type checking since the types
    // contain TypeParamType which will be resolved during monomorphization
    return success();
  }

  // Regular function call (no type arguments)
  // Check for FuncOp first
  auto funcOp =
      symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, getCalleeAttr());
  if (!funcOp) {
    // Check for SpecializedFuncOp (after monomorphization)
    auto specializedOp = symbolTable.lookupNearestSymbolFrom<SpecializedFuncOp>(
        *this, getCalleeAttr());
    if (specializedOp) {
      // Specialized functions are valid call targets - detailed type checking
      // is done during lowering when the body is instantiated
      return success();
    }
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

//===----------------------------------------------------------------------===//
// GenericFuncOp
//===----------------------------------------------------------------------===//

void GenericFuncOp::print(OpAsmPrinter& p) {
  p << " @" << getSymName() << "<";

  // Print type parameters with constraints
  auto typeParams = getTypeParams();
  auto constraints = getConstraints();
  for (size_t i = 0; i < typeParams.size(); ++i) {
    if (i > 0) {
      p << ", ";
    }
    auto name = llvm::cast<StringAttr>(typeParams[i]).getValue();
    p << name;
    auto constraint = llvm::cast<TypeParamKindAttr>(constraints[i]).getValue();
    if (constraint != TypeParamKind::Any) {
      p << ": " << stringifyTypeParamKind(constraint);
    }
  }
  p << ">";

  // Print function signature using MLIR's function printing utility
  function_interface_impl::printFunctionSignature(
      p, *this, getFunctionType().getInputs(), /*isVariadic=*/false,
      getFunctionType().getResults());

  // Print body
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);

  // Print optional attributes
  function_interface_impl::printFunctionAttributes(
      p, *this,
      {getSymNameAttrName(), getFunctionTypeAttrName(), getTypeParamsAttrName(),
       getConstraintsAttrName(), getArgAttrsAttrName(), getResAttrsAttrName()});
}

ParseResult GenericFuncOp::parse(OpAsmParser& parser, OperationState& result) {
  StringAttr nameAttr;
  SmallVector<Attribute> typeParamAttrs;
  SmallVector<Attribute> constraintAttrs;

  // Parse function name
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name),
                             result.attributes)) {
    return failure();
  }

  // Parse type parameters: <T, N: numeric, ...>
  if (parser.parseLess()) {
    return failure();
  }

  do {
    StringRef paramName;
    if (parser.parseKeyword(&paramName)) {
      return failure();
    }
    typeParamAttrs.push_back(parser.getBuilder().getStringAttr(paramName));

    // Parse optional constraint
    TypeParamKind constraint = TypeParamKind::Any;
    if (succeeded(parser.parseOptionalColon())) {
      StringRef constraintStr;
      if (parser.parseKeyword(&constraintStr)) {
        return failure();
      }
      auto maybeConstraint = symbolizeTypeParamKind(constraintStr);
      if (!maybeConstraint) {
        return parser.emitError(parser.getCurrentLocation(),
                                "invalid type parameter constraint");
      }
      constraint = *maybeConstraint;
    }
    constraintAttrs.push_back(
        TypeParamKindAttr::get(parser.getContext(), constraint));
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater()) {
    return failure();
  }

  result.addAttribute(getTypeParamsAttrName(result.name),
                      parser.getBuilder().getArrayAttr(typeParamAttrs));
  result.addAttribute(getConstraintsAttrName(result.name),
                      parser.getBuilder().getArrayAttr(constraintAttrs));

  // Parse function signature
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  bool isVariadic = false;

  if (function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, args, isVariadic, resultTypes,
          resultAttrs)) {
    return failure();
  }

  SmallVector<Type> argTypes;
  for (auto& arg : args) {
    argTypes.push_back(arg.type);
  }
  auto funcType = parser.getBuilder().getFunctionType(argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(funcType));

  // Parse function body
  auto* body = result.addRegion();
  if (parser.parseRegion(*body, args)) {
    return failure();
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return failure();
  }

  return success();
}

LogicalResult GenericFuncOp::verify() {
  auto typeParams = getTypeParams();
  auto constraints = getConstraints();

  // Check that we have at least one type parameter
  if (typeParams.empty()) {
    return emitOpError(
        "generic function must have at least one type parameter");
  }

  // Check that type_params and constraints have the same size
  if (typeParams.size() != constraints.size()) {
    return emitOpError("type_params and constraints must have the same size");
  }

  // Check for duplicate type parameter names
  llvm::SmallDenseSet<llvm::StringRef> seenNames;
  for (auto param : typeParams) {
    auto name = llvm::cast<StringAttr>(param).getValue();
    if (!seenNames.insert(name).second) {
      return emitOpError("duplicate type parameter name '") << name << "'";
    }
  }

  // Collect all declared type parameter names
  llvm::SmallDenseSet<llvm::StringRef> declaredParams;
  for (auto param : typeParams) {
    declaredParams.insert(llvm::cast<StringAttr>(param).getValue());
  }

  // Check that all TypeParamTypes in the function signature reference declared
  // type parameters
  auto funcType = getFunctionType();
  auto checkType = [&](Type type) -> LogicalResult {
    if (auto paramType = llvm::dyn_cast<TypeParamType>(type)) {
      if (!declaredParams.contains(paramType.getName())) {
        return emitOpError("use of undeclared type parameter '")
               << paramType.getName() << "'";
      }
    }
    return success();
  };

  for (Type inputType : funcType.getInputs()) {
    if (failed(checkType(inputType))) {
      return failure();
    }
  }
  for (Type resultType : funcType.getResults()) {
    if (failed(checkType(resultType))) {
      return failure();
    }
  }

  return success();
}

llvm::SmallVector<llvm::StringRef> GenericFuncOp::getTypeParamNames() {
  llvm::SmallVector<llvm::StringRef> names;
  for (auto param : getTypeParams()) {
    names.push_back(llvm::cast<StringAttr>(param).getValue());
  }
  return names;
}

TypeParamKind GenericFuncOp::getConstraintAt(unsigned index) {
  return llvm::cast<TypeParamKindAttr>(getConstraints()[index]).getValue();
}

std::optional<TypeParamKind>
GenericFuncOp::getConstraintFor(llvm::StringRef name) {
  auto typeParams = getTypeParams();
  for (size_t i = 0; i < typeParams.size(); ++i) {
    if (llvm::cast<StringAttr>(typeParams[i]).getValue() == name) {
      return getConstraintAt(i);
    }
  }
  return std::nullopt;
}

bool GenericFuncOp::hasTypeParam(llvm::StringRef name) {
  for (auto param : getTypeParams()) {
    if (llvm::cast<StringAttr>(param).getValue() == name) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// SpecializedFuncOp
//===----------------------------------------------------------------------===//

LogicalResult SpecializedFuncOp::verify() {
  // Look up the generic function
  auto* parentOp = (*this)->getParentOp();
  auto* symbolTableOp = parentOp;
  while (symbolTableOp && !symbolTableOp->hasTrait<OpTrait::SymbolTable>()) {
    symbolTableOp = symbolTableOp->getParentOp();
  }

  if (!symbolTableOp) {
    return emitOpError("could not find symbol table");
  }

  auto genericOp =
      SymbolTable::lookupSymbolIn(symbolTableOp, getGenericFuncAttr());
  if (!genericOp) {
    return emitOpError("references undefined generic function '")
           << getGenericFunc() << "'";
  }

  auto genericFuncOp = dyn_cast<GenericFuncOp>(genericOp);
  if (!genericFuncOp) {
    return emitOpError("'")
           << getGenericFunc() << "' is not a generic function";
  }

  // Check type argument count matches type parameter count
  auto typeArgs = getTypeArgs();
  unsigned numTypeParams = genericFuncOp.getNumTypeParams();
  if (typeArgs.size() != numTypeParams) {
    return emitOpError("expects ")
           << numTypeParams << " type argument(s) but got " << typeArgs.size();
  }

  // Check that each type argument satisfies its constraint
  for (size_t i = 0; i < typeArgs.size(); ++i) {
    auto typeArg = llvm::cast<TypeAttr>(typeArgs[i]).getValue();
    auto constraint = genericFuncOp.getConstraintAt(i);
    auto paramName =
        llvm::cast<StringAttr>(genericFuncOp.getTypeParams()[i]).getValue();

    // Check constraint satisfaction
    bool satisfies = false;
    switch (constraint) {
    case TypeParamKind::Any:
      satisfies = llvm::isa<polang::IntegerType>(typeArg) ||
                  llvm::isa<polang::FloatType>(typeArg) ||
                  llvm::isa<BoolType>(typeArg);
      break;
    case TypeParamKind::Numeric:
      satisfies = llvm::isa<polang::IntegerType>(typeArg) ||
                  llvm::isa<polang::FloatType>(typeArg);
      break;
    case TypeParamKind::Integer:
      satisfies = llvm::isa<polang::IntegerType>(typeArg);
      break;
    case TypeParamKind::Float:
      satisfies = llvm::isa<polang::FloatType>(typeArg);
      break;
    }

    if (!satisfies) {
      return emitOpError("type argument ")
             << typeArg << " does not satisfy '"
             << stringifyTypeParamKind(constraint)
             << "' constraint for type parameter '" << paramName << "'";
    }
  }

  return success();
}

FunctionType SpecializedFuncOp::getSpecializedType(GenericFuncOp genericOp) {
  auto typeArgs = getTypeArgs();
  auto genericType = genericOp.getFunctionType();

  // Build a mapping from type parameter names to concrete types
  llvm::StringMap<Type> typeMap;
  auto typeParams = genericOp.getTypeParams();
  for (size_t i = 0; i < typeParams.size(); ++i) {
    auto name = llvm::cast<StringAttr>(typeParams[i]).getValue();
    auto concreteType = llvm::cast<TypeAttr>(typeArgs[i]).getValue();
    typeMap[name] = concreteType;
  }

  // Substitute types in function signature
  auto substituteType = [&](Type type) -> Type {
    if (auto paramType = llvm::dyn_cast<TypeParamType>(type)) {
      auto it = typeMap.find(paramType.getName());
      if (it != typeMap.end()) {
        return it->second;
      }
    }
    return type;
  };

  SmallVector<Type> inputTypes;
  for (Type inputType : genericType.getInputs()) {
    inputTypes.push_back(substituteType(inputType));
  }

  SmallVector<Type> resultTypes;
  for (Type resultType : genericType.getResults()) {
    resultTypes.push_back(substituteType(resultType));
  }

  return FunctionType::get(getContext(), inputTypes, resultTypes);
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
    if (parser.parseRegion(*bindingRegion, /*arguments=*/{}, /*argTypes=*/{})) {
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
