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
