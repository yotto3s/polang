add_mlir_conversion_library(ASTToPolang
  ASTToPolang.cpp

  DEPENDS
  PolangASTDialect
  PolangDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  PolangASTDialect
  PolangDialect
)
