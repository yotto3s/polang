add_mlir_library(PolangMLIRGen
  MLIRGen.cpp
  PolangTypeConverter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/polang
  ${PROJECT_BINARY_DIR}/parser  # For generated parser.hpp

  DEPENDS
  PolangOpsIncGen

  LINK_LIBS PUBLIC
  PolangDialect
  PolangParser
  MLIRIR
  MLIRSupport
)

# Add include directory for generated parser header
target_include_directories(obj.PolangMLIRGen PRIVATE
  ${PROJECT_BINARY_DIR}/parser
)
