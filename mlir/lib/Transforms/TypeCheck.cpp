//===- TypeCheck.cpp - Type checking pass -----------------------*- C++ -*-===//
//
// This file implements the type checking pass for the Polang AST dialect.
// It validates semantic correctness after name resolution.
//
//===----------------------------------------------------------------------===//

// Suppress warnings from MLIR/LLVM headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "polang/Dialect/PolangASTDialect.h"
#include "polang/Dialect/PolangASTOps.h"
#include "polang/Dialect/PolangASTTypes.h"
#include "polang/Dialect/PolangTypes.h"
#include "polang/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#pragma GCC diagnostic pop

using namespace mlir;

namespace {

/// Check if a type is a TypeVarType
[[nodiscard]] bool isTypeVar(Type type) {
  return mlir::isa<polang::ast::TypeVarType>(type);
}

/// Get a printable name for a type (for error messages)
[[nodiscard]] std::string getTypeName(Type type) {
  if (auto intType = mlir::dyn_cast<polang::IntegerType>(type)) {
    return "i" + std::to_string(intType.getWidth());
  }
  if (mlir::isa<polang::BoolType>(type)) {
    return "bool";
  }
  if (mlir::isa<polang::FloatType>(type)) {
    return "float";
  }
  if (auto typeVar = mlir::dyn_cast<polang::ast::TypeVarType>(type)) {
    return "typevar<" + std::to_string(typeVar.getId()) + ">";
  }
  // Default: use MLIR's string representation
  std::string result;
  llvm::raw_string_ostream os(result);
  type.print(os);
  return result;
}

/// Check if two concrete types are compatible for binary operations
[[nodiscard]] bool areTypesCompatible(Type lhs, Type rhs) {
  // Type variables are always compatible (inference handles them)
  if (isTypeVar(lhs) || isTypeVar(rhs)) {
    return true;
  }
  // Otherwise, types must be equal
  return lhs == rhs;
}

//===----------------------------------------------------------------------===//
// TypeCheckPass
//===----------------------------------------------------------------------===//

struct TypeCheckPass
    : public PassWrapper<TypeCheckPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeCheckPass)

  [[nodiscard]] StringRef getArgument() const override {
    return "polang-type-check";
  }

  [[nodiscard]] StringRef getDescription() const override {
    return "Validate type correctness of AST dialect IR";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<polang::ast::PolangASTDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    bool hadError = false;

    // Walk all operations
    module.walk([&](Operation* op) {
      // Check binary operations (add, sub, mul, div)
      if (auto addOp = dyn_cast<polang::ast::AddOp>(op)) {
        checkBinaryOp(addOp.getLhs().getType(), addOp.getRhs().getType(), op,
                      hadError);
      } else if (auto subOp = dyn_cast<polang::ast::SubOp>(op)) {
        checkBinaryOp(subOp.getLhs().getType(), subOp.getRhs().getType(), op,
                      hadError);
      } else if (auto mulOp = dyn_cast<polang::ast::MulOp>(op)) {
        checkBinaryOp(mulOp.getLhs().getType(), mulOp.getRhs().getType(), op,
                      hadError);
      } else if (auto divOp = dyn_cast<polang::ast::DivOp>(op)) {
        checkBinaryOp(divOp.getLhs().getType(), divOp.getRhs().getType(), op,
                      hadError);
      }

      // Check if condition type
      if (auto ifOp = dyn_cast<polang::ast::IfOp>(op)) {
        checkIfCondition(ifOp, hadError);
      }

      // Check function call arity
      if (auto callOp = dyn_cast<polang::ast::CallOp>(op)) {
        checkCallArity(callOp, module, hadError);
      }
    });

    if (hadError) {
      signalPassFailure();
    }
  }

private:
  /// Check binary operation type compatibility
  void checkBinaryOp(Type lhsType, Type rhsType, Operation* op,
                     bool& hadError) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      op->emitError("binary operation type mismatch: ")
          << getTypeName(lhsType) << " vs " << getTypeName(rhsType);
      hadError = true;
    }
  }

  /// Check if condition must be bool or unconstrained type variable
  void checkIfCondition(polang::ast::IfOp ifOp, bool& hadError) {
    Type condType = ifOp.getCondition().getType();

    // Bool is always valid
    if (mlir::isa<polang::BoolType>(condType)) {
      return;
    }

    // Unconstrained type variable is valid (inference will resolve)
    if (auto typeVar = mlir::dyn_cast<polang::ast::TypeVarType>(condType)) {
      if (typeVar.getKind() == polang::TypeVarKind::Any) {
        return;
      }
      // Integer or float type variables are NOT compatible with bool
      ifOp.emitError("if condition type must be bool or unconstrained type "
                     "variable, got '")
          << condType << "'";
      hadError = true;
      return;
    }

    // Any other type is invalid
    ifOp.emitError("if condition must be bool, got '") << condType << "'";
    hadError = true;
  }

  /// Check function call arity
  void checkCallArity(polang::ast::CallOp callOp, mlir::ModuleOp module,
                      bool& hadError) {
    auto callee = callOp.getCallee();
    auto numArgs = callOp.getOperands().size();

    // Look up the function
    auto funcOp = module.lookupSymbol<polang::ast::FuncOp>(callee);
    if (!funcOp) {
      // Function not found - let other passes handle this error
      return;
    }

    auto expectedArgs = funcOp.getNumArguments();
    if (numArgs != expectedArgs) {
      callOp.emitError("function @") << callee << " expects " << expectedArgs
                                     << " arguments but got " << numArgs;
      hadError = true;
    }
  }
};

} // namespace

namespace polang {

std::unique_ptr<Pass> createTypeCheckPass() {
  return std::make_unique<TypeCheckPass>();
}

} // namespace polang
