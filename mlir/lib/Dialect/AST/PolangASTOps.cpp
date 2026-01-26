//===- PolangASTOps.cpp - Polang AST operation implementation ----*- C++ -*-===//
//
// This file implements the operations for the Polang AST dialect.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers and generated code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangASTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangASTTypes.h"
#include "polang/Dialect/PolangTypes.h"

using namespace mlir;
using namespace polang::ast;

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
  if (isa<polang::TypeVarType>(t1) || isa<polang::TypeVarType>(t2)) {
    return true;
  }
  return false;
}

/// Check if a type is compatible with boolean for if conditions.
/// A type is boolean-compatible if it's:
/// - polang.bool (concrete bool type)
/// - A type variable with 'any' kind (can unify with bool)
/// Type variables with 'integer' or 'float' kind are NOT compatible.
bool isBoolCompatible(Type type) {
  if (isa<polang::BoolType>(type)) {
    return true;
  }
  if (auto typeVar = dyn_cast<TypeVarType>(type)) {
    return typeVar.isAny();
  }
  if (auto typeVar = dyn_cast<polang::TypeVarType>(type)) {
    return typeVar.isAny();
  }
  return false;
}
} // namespace

#define GET_OP_CLASSES
#include "polang/Dialect/PolangASTOps.cpp.inc"

#pragma GCC diagnostic pop

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
  } else if (!isa<TypeVarType>(resultType) &&
             !isa<polang::TypeVarType>(resultType)) {
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
  } else if (!isa<TypeVarType>(resultType) &&
             !isa<polang::TypeVarType>(resultType)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected polang.float or typevar type");
  }
  auto attr = FloatAttr::get(attrType, value);
  result.addAttribute("value", attr);
  result.addTypes(resultType);
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
  if (isa<polang::TypeVarType>(inputType) ||
      isa<polang::TypeVarType>(resultType)) {
    return success();
  }

  // Both types must be numeric
  const bool inputIsNumeric = isa<polang::IntegerType>(inputType) ||
                              isa<polang::FloatType>(inputType);
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
// CmpOp verifier
//===----------------------------------------------------------------------===//

LogicalResult CmpOp::verify() {
  if (!typesAreCompatible(getLhs().getType(), getRhs().getType())) {
    return emitOpError("comparison operand types must be compatible");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp builder and verifier
//===----------------------------------------------------------------------===//

void IfOp::build(OpBuilder& builder, OperationState& state, Type resultType,
                 Value condition) {
  (void)builder;
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
  // Check condition type is compatible with boolean
  Type condType = getCondition().getType();
  if (!isBoolCompatible(condType)) {
    return emitOpError(
               "condition type must be bool or unconstrained type variable, got ")
           << condType;
  }

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
    return emitOpError("then region must end with polang_ast.yield");
  }
  if (!elseYield) {
    return emitOpError("else region must end with polang_ast.yield");
  }

  // Check that yield types are compatible with result type
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
// FuncOp builder, verifier, and custom print/parse
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder& builder, OperationState& state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty()) {
    return;
  }
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

void FuncOp::print(OpAsmPrinter& p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

ParseResult FuncOp::parse(OpAsmParser& parser, OperationState& result) {
  auto buildFuncType = [](Builder& builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string&) {
    return builder.getFunctionType(argTypes, results);
  };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

LogicalResult FuncOp::verify() {
  // Function must have a body
  if (getBody().empty()) {
    return emitOpError("must have a body");
  }

  // Verify the function type matches the entry block arguments

  FunctionType funcType = getFunctionType();
  Block& entryBlock = getBody().front();

  if (funcType.getNumInputs() != entryBlock.getNumArguments()) {
    return emitOpError("function type has ")
           << funcType.getNumInputs() << " arguments but entry block has "
           << entryBlock.getNumArguments();
  }

  for (unsigned i = 0, e = funcType.getNumInputs(); i != e; ++i) {
    if (funcType.getInput(i) != entryBlock.getArgument(i).getType()) {
      return emitOpError("type of entry block argument #")
             << i << " (" << entryBlock.getArgument(i).getType()
             << ") must match the type of the function argument ("
             << funcType.getInput(i) << ")";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp interface methods and verifySymbolUses
//===----------------------------------------------------------------------===//

CallInterfaceCallable CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", llvm::cast<SymbolRefAttr>(callee));
}

Operation::operand_range CallOp::getArgOperands() {
  return getOperands();
}

MutableOperandRange CallOp::getArgOperandsMutable() {
  return getOperandsMutable();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr) {
    return emitOpError("requires a 'callee' symbol reference attribute");
  }

  auto fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn) {
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";
  }

  // Verify that the operand and result types match the callee.
  FunctionType fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands()) {
    return emitOpError("incorrect number of operands for callee");
  }

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    if (!typesAreCompatible(getOperand(i).getType(), fnType.getInput(i))) {
      return emitOpError("operand type mismatch: expected ")
             << fnType.getInput(i) << " but got " << getOperand(i).getType()
             << " for operand #" << i;
    }
  }

  if (fnType.getNumResults() != getNumResults()) {
    return emitOpError("incorrect number of results for callee");
  }

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    if (!typesAreCompatible(getResult(i).getType(), fnType.getResult(i))) {
      return emitOpError("result type mismatch: expected ")
             << fnType.getResult(i) << " but got " << getResult(i).getType()
             << " for result #" << i;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LetExprOp custom print/parse and verifier
//===----------------------------------------------------------------------===//

void LetExprOp::print(OpAsmPrinter& p) {
  // Print var_names attribute
  p << " ";
  p.printAttribute(getVarNamesAttr());
  p << " -> " << getResult().getType();

  // Print each binding region
  for (Region& binding : getBindings()) {
    p << "\n  binding ";
    p.printRegion(binding, /*printEntryBlockArgs=*/false);
  }

  // Print body region
  p << "\n  body ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"var_names"});
}

ParseResult LetExprOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse: ["name1", "name2", ...] -> type binding { } binding { } body { }
  ArrayAttr varNamesAttr;
  Type resultType;

  if (parser.parseAttribute(varNamesAttr, "var_names", result.attributes) ||
      parser.parseArrow() || parser.parseType(resultType)) {
    return failure();
  }
  result.addTypes(resultType);

  // Note: Body region must be added first (MLIR requires variadic regions last)
  // but we parse bindings first for natural reading order

  // Parse binding regions into temporary storage
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

  // Parse body region and add it first (as required by region order)
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

  // Now add binding regions (variadic, must be last in region list)
  for (auto& bindingRegion : bindingRegions) {
    Region* binding = result.addRegion();
    binding->takeBody(*bindingRegion);
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

LogicalResult LetExprOp::verify() {
  // 1. Must have at least one binding
  if (getBindings().empty()) {
    return emitOpError("must have at least one binding region");
  }

  // 2. Number of binding regions must match var_names count
  if (getBindings().size() != getVarNames().size()) {
    return emitOpError("number of binding regions (")
           << getBindings().size() << ") must match var_names count ("
           << getVarNames().size() << ")";
  }

  // 3. Each binding region must have exactly one block ending with yield.binding
  SmallVector<Type> bindingTypes;
  for (auto [idx, binding] : llvm::enumerate(getBindings())) {
    if (binding.empty() || binding.front().empty()) {
      return emitOpError("binding region #") << idx << " must not be empty";
    }

    auto* terminator = binding.front().getTerminator();
    auto yieldBinding = dyn_cast<YieldBindingOp>(terminator);
    if (!yieldBinding) {
      return emitOpError("binding region #")
             << idx << " must end with polang_ast.yield.binding";
    }
    bindingTypes.push_back(yieldBinding.getValue().getType());
  }

  // 4. Body region must not be empty
  if (getBody().empty() || getBody().front().empty()) {
    return emitOpError("body region must not be empty");
  }

  // 5. Body block arguments must match binding yields
  Block& bodyEntry = getBody().front();
  if (bodyEntry.getNumArguments() != bindingTypes.size()) {
    return emitOpError("body block argument count (")
           << bodyEntry.getNumArguments()
           << ") doesn't match binding count (" << bindingTypes.size() << ")";
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

  // 6. Body must end with polang_ast.yield
  auto* bodyTerminator = getBody().front().getTerminator();
  auto yieldOp = dyn_cast<YieldOp>(bodyTerminator);
  if (!yieldOp) {
    return emitOpError("body must end with polang_ast.yield");
  }

  // 7. Yield type must match result type
  if (!typesAreCompatible(yieldOp.getValue().getType(), getResult().getType())) {
    return emitOpError("yield type ")
           << yieldOp.getValue().getType() << " doesn't match result type "
           << getResult().getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ImportOp custom print/parse
//===----------------------------------------------------------------------===//

void ImportOp::print(OpAsmPrinter& p) {
  p << " ";
  p.printAttribute(getModuleNameAttr());
  if (getAlias()) {
    p << " as ";
    p.printAttribute(getAliasAttr());
  }
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"module_name", "alias"});
}

ParseResult ImportOp::parse(OpAsmParser& parser, OperationState& result) {
  FlatSymbolRefAttr moduleRef;
  if (parser.parseAttribute(moduleRef, "module_name", result.attributes)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("as"))) {
    FlatSymbolRefAttr aliasRef;
    if (parser.parseAttribute(aliasRef, "alias", result.attributes)) {
      return failure();
    }
  }

  return parser.parseOptionalAttrDict(result.attributes);
}
