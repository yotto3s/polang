# TableGen for Polang dialect
# Ensure output directory exists before running TableGen
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/polang/Dialect)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/polang/Transforms)

# Add MLIR include directories for TableGen
# The tablegen() function uses get_directory_property(INCLUDE_DIRECTORIES)
# to find .td files, so we must add includes here
# This does NOT affect GoogleTest since it's in a separate directory
include_directories(${POLANG_MLIR_INCLUDE_DIRS})
include_directories(${POLANG_LLVM_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS include/polang/Dialect/PolangOps.td)
mlir_tablegen(include/polang/Dialect/PolangOps.h.inc -gen-op-decls)
mlir_tablegen(include/polang/Dialect/PolangOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/polang/Dialect/PolangDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/polang/Dialect/PolangDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(include/polang/Dialect/PolangEnums.h.inc -gen-enum-decls)
mlir_tablegen(include/polang/Dialect/PolangEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(PolangOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/polang/Dialect/PolangTypes.td)
mlir_tablegen(include/polang/Dialect/PolangTypes.h.inc -gen-typedef-decls)
mlir_tablegen(include/polang/Dialect/PolangTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(PolangTypesIncGen)

set(LLVM_TARGET_DEFINITIONS include/polang/Transforms/Passes.td)
mlir_tablegen(include/polang/Transforms/Passes.h.inc -gen-pass-decls -name PolangTransforms)
add_public_tablegen_target(PolangTransformsPassIncGen)

# AST Dialect TableGen
set(LLVM_TARGET_DEFINITIONS include/polang/Dialect/PolangASTOps.td)
mlir_tablegen(include/polang/Dialect/PolangASTOps.h.inc -gen-op-decls -dialect=polang_ast)
mlir_tablegen(include/polang/Dialect/PolangASTOps.cpp.inc -gen-op-defs -dialect=polang_ast)
mlir_tablegen(include/polang/Dialect/PolangASTDialect.h.inc -gen-dialect-decls -dialect=polang_ast)
mlir_tablegen(include/polang/Dialect/PolangASTDialect.cpp.inc -gen-dialect-defs -dialect=polang_ast)
add_public_tablegen_target(PolangASTOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/polang/Dialect/PolangASTTypes.td)
mlir_tablegen(include/polang/Dialect/PolangASTTypes.h.inc -gen-typedef-decls --typedefs-dialect=polang_ast)
mlir_tablegen(include/polang/Dialect/PolangASTTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=polang_ast)
add_public_tablegen_target(PolangASTTypesIncGen)

add_subdirectory(lib/Dialect)
add_subdirectory(lib/Conversion)
add_subdirectory(lib/MLIRGen)
add_subdirectory(lib/Transforms)
add_subdirectory(tools)

# Combined MLIR library
add_library(PolangMLIR INTERFACE)
target_link_libraries(PolangMLIR INTERFACE
  PolangDialect
  PolangASTDialect
  PolangConversion
  PolangMLIRGen
  PolangTransforms
)
