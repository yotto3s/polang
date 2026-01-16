# REPL integration test
add_executable(repl_test repl_test.cpp)
target_include_directories(repl_test PRIVATE ${POLANG_TEST_COMMON_DIR})
target_link_libraries(repl_test PRIVATE GTest::gtest_main GTest::gmock)
set_target_properties(repl_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin"
)
target_compile_definitions(repl_test PRIVATE
    POLANG_COMPILER_PATH="$<TARGET_FILE:PolangCompiler>"
    POLANG_REPL_PATH="$<TARGET_FILE:PolangRepl>"
)
target_compile_options(repl_test PRIVATE -Wno-error)
gtest_discover_tests(repl_test)

# REPL unit tests (InputChecker - header-only, no LLVM deps)
add_executable(repl_unit_test repl_unit_test.cpp)
target_include_directories(repl_unit_test PRIVATE ${CMAKE_SOURCE_DIR}/repl/include)
target_link_libraries(repl_unit_test PRIVATE GTest::gtest_main)
set_target_properties(repl_unit_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin"
)
target_compile_options(repl_unit_test PRIVATE -Wno-error)
gtest_discover_tests(repl_unit_test)
