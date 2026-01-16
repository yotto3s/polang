// Stub implementations of codeGen methods for testing without LLVM codegen
// These are overridden by the real implementations in compiler/src/codegen.cpp
// when linking with PolangCompiler

#include "parser/node.hpp"

// All codeGen methods return nullptr - actual implementations are in codegen.cpp
llvm::Value *NInteger::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NDouble::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NIdentifier::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NMethodCall::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NBinaryOperator::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NAssignment::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NBlock::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NIfExpression::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NExpressionStatement::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NVariableDeclaration::codeGen(CodeGenContext &context) { return nullptr; }
llvm::Value *NFunctionDeclaration::codeGen(CodeGenContext &context) { return nullptr; }
