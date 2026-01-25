# Lit test configuration for Polang

# Find Python (required for lit)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Lit executable path (LLVM's bundled lit.py)
set(LIT_COMMAND "${Python3_EXECUTABLE}" "/usr/lib/llvm-20/build/utils/lit/lit.py"
    CACHE STRING "Command to run lit")

# LLVM tools paths
set(LLVM_TOOLS_DIR "/usr/lib/llvm-20/bin" CACHE PATH "Path to LLVM tools")
set(FILECHECK_PATH "${LLVM_TOOLS_DIR}/FileCheck" CACHE FILEPATH "Path to FileCheck")
set(NOT_PATH "${LLVM_TOOLS_DIR}/not" CACHE FILEPATH "Path to not tool")

# Configure paths for lit.site.cfg.py
set(POLANG_OBJ_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(POLANG_SRC_ROOT "${CMAKE_SOURCE_DIR}")
set(POLANG_COMPILER_PATH "${CMAKE_BINARY_DIR}/bin/PolangCompiler")
set(POLANG_REPL_PATH "${CMAKE_BINARY_DIR}/bin/PolangRepl")
set(POLANG_OPT_PATH "${CMAKE_BINARY_DIR}/bin/polang-opt")

# Generate lit.site.cfg.py from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
    @ONLY
)

# Copy lit.cfg.py to build directory
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
    "${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py"
    COPYONLY
)

# Copy test files to build directory
set(TEST_SUBDIRS AST MLIR LLVMIR Execution Errors)
foreach(SUBDIR ${TEST_SUBDIRS})
    file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/*.po")
    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME)
        configure_file(
            "${TEST_FILE}"
            "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${TEST_NAME}"
            COPYONLY
        )
    endforeach()
endforeach()

# Copy MLIR AST dialect test files
file(GLOB MLIR_AST_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/MLIR/AST/*.mlir")
foreach(TEST_FILE ${MLIR_AST_TEST_FILES})
    get_filename_component(TEST_NAME "${TEST_FILE}" NAME)
    configure_file(
        "${TEST_FILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/MLIR/AST/${TEST_NAME}"
        COPYONLY
    )
endforeach()

# Add custom target to run lit tests
add_custom_target(check-polang-lit
    COMMAND ${LIT_COMMAND} -v "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS PolangCompiler PolangRepl polang-opt
    COMMENT "Running Polang lit tests"
    USES_TERMINAL
)

# Add test for ctest integration
add_test(
    NAME polang-lit-tests
    COMMAND ${LIT_COMMAND} -v "${CMAKE_CURRENT_BINARY_DIR}"
)

# Set dependencies for the test
set_tests_properties(polang-lit-tests PROPERTIES
    DEPENDS "PolangCompiler;PolangRepl;polang-opt"
)
