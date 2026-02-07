add_mlir_library(PolangTransforms
  Monomorphization.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/polang/Transforms

  DEPENDS
  PolangTransformsPassIncGen

  LINK_LIBS PUBLIC
  PolangDialect
  MLIRIR
  MLIRPass
  MLIRFuncDialect
)
