add_mlir_library(PolangTransforms
  NameResolution.cpp
  TypeCheck.cpp
  TypeInference.cpp
  Monomorphization.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/polang/Transforms

  DEPENDS
  PolangTransformsPassIncGen

  LINK_LIBS PUBLIC
  PolangDialect
  PolangASTDialect
  MLIRIR
  MLIRPass
  MLIRFuncDialect
)
