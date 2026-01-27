add_mlir_library(PolangConversion
  PolangToStandard.cpp
  ASTToPolang/ASTToPolang.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/polang/Conversion

  DEPENDS
  PolangOpsIncGen
  PolangASTOpsIncGen

  LINK_LIBS PUBLIC
  PolangDialect
  PolangASTDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRPass
)

# Suppress overloaded-virtual warning for MLIR patterns
target_compile_options(obj.PolangConversion PRIVATE -Wno-overloaded-virtual)
