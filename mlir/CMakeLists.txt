# TableGen for Polang dialect
# Ensure output directory exists before running TableGen
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/polang/Dialect)

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

add_subdirectory(lib/Dialect)
add_subdirectory(lib/Conversion)
add_subdirectory(lib/MLIRGen)

# Combined MLIR library
add_library(PolangMLIR INTERFACE)
target_link_libraries(PolangMLIR INTERFACE
  PolangDialect
  PolangConversion
  PolangMLIRGen
)
