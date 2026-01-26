add_mlir_dialect_library(PolangASTDialect
  PolangASTDialect.cpp
  PolangASTOps.cpp
  PolangASTTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/polang/Dialect

  DEPENDS
  PolangASTOpsIncGen
  PolangASTTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  PolangDialect
)
