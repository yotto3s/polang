# Compiler component tests
add_executable(compiler_test compiler_test.cpp)
target_include_directories(compiler_test PRIVATE ${POLANG_TEST_COMMON_DIR})
target_link_libraries(compiler_test PRIVATE GTest::gtest_main GTest::gmock)
set_target_properties(compiler_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin"
)
target_compile_definitions(compiler_test PRIVATE
    POLANG_COMPILER_PATH="$<TARGET_FILE:PolangCompiler>"
    POLANG_REPL_PATH="$<TARGET_FILE:PolangRepl>"
)
target_compile_options(compiler_test PRIVATE -Wno-error)
gtest_discover_tests(compiler_test)

# MLIR verifier unit tests (directly construct invalid MLIR)
# Note: When using Clang, LLVM's cxxabi.h conflicts with libstdc++.
# We exclude LLVM's root include from the search path and use only llvm/ and mlir/ subdirs.
add_executable(mlir_verifier_test mlir_verifier_test.cpp)
target_include_directories(mlir_verifier_test PRIVATE
    ${PROJECT_SOURCE_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/mlir/include
)
# Use --include-directory-after for LLVM to ensure system stdlib is searched first
target_compile_options(mlir_verifier_test PRIVATE
    -Wno-error -Wno-unused-parameter -Wno-overloaded-virtual
    "SHELL:-idirafter ${POLANG_LLVM_INCLUDE_DIRS}"
)
target_compile_definitions(mlir_verifier_test PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_directories(mlir_verifier_test PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(mlir_verifier_test PRIVATE GTest::gtest_main)
target_link_libraries(mlir_verifier_test PRIVATE
    $<TARGET_FILE:PolangDialect>
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRPass
)
add_dependencies(mlir_verifier_test PolangDialect PolangOpsIncGen PolangTypesIncGen)
set_target_properties(mlir_verifier_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin"
)
gtest_discover_tests(mlir_verifier_test)

# TypeInference pass unit tests (exercise error paths in TypeInference.cpp)
add_executable(type_inference_pass_test type_inference_pass_test.cpp)
target_include_directories(type_inference_pass_test PRIVATE
    ${PROJECT_SOURCE_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/mlir/include
)
target_compile_options(type_inference_pass_test PRIVATE
    -Wno-error -Wno-unused-parameter -Wno-overloaded-virtual
    "SHELL:-idirafter ${POLANG_LLVM_INCLUDE_DIRS}"
)
target_compile_definitions(type_inference_pass_test PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_directories(type_inference_pass_test PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(type_inference_pass_test PRIVATE GTest::gtest_main)
target_link_libraries(type_inference_pass_test PRIVATE
    $<TARGET_FILE:PolangTransforms>
    $<TARGET_FILE:PolangDialect>
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRPass
)
add_dependencies(type_inference_pass_test PolangDialect PolangTransforms PolangOpsIncGen PolangTypesIncGen PolangTransformsPassIncGen)
set_target_properties(type_inference_pass_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin"
)
gtest_discover_tests(type_inference_pass_test)

# PolangToStandard conversion pass unit tests (AllocaOp, PrintOp lowering)
add_executable(conversion_pass_test conversion_pass_test.cpp)
target_include_directories(conversion_pass_test PRIVATE
    ${PROJECT_SOURCE_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/mlir/include
)
target_compile_options(conversion_pass_test PRIVATE
    -Wno-error -Wno-unused-parameter -Wno-overloaded-virtual
    "SHELL:-idirafter ${POLANG_LLVM_INCLUDE_DIRS}"
)
target_compile_definitions(conversion_pass_test PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_directories(conversion_pass_test PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(conversion_pass_test PRIVATE GTest::gtest_main)
target_link_libraries(conversion_pass_test PRIVATE
    $<TARGET_FILE:PolangConversion>
    $<TARGET_FILE:PolangDialect>
    MLIRIR
    MLIRSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRTransforms
    MLIRPass
)
add_dependencies(conversion_pass_test PolangDialect PolangConversion PolangOpsIncGen PolangTypesIncGen)
set_target_properties(conversion_pass_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin"
)
gtest_discover_tests(conversion_pass_test)
