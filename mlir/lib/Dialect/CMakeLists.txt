add_mlir_dialect_library(PolangDialect
  PolangDialect.cpp
  PolangOps.cpp
  PolangTypes.cpp
  PolangTypeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/polang/Dialect

  DEPENDS
  PolangOpsIncGen
  PolangTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
)

add_subdirectory(AST)
