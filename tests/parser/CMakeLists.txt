# List of test names (each corresponds to <name>_test.cpp)
set(PARSER_TESTS
    lexer
    parser_declaration
    parser_expression
    parser_control_flow
    parser_statement
    error
    type_check
    ast_printer
)

foreach(test_name ${PARSER_TESTS})
    set(test_target ${test_name}_test)
    add_executable(${test_target} ${test_target}.cpp)
    target_link_libraries(${test_target}
        PRIVATE PolangParser
        PRIVATE GTest::gtest_main
    )
    set_target_properties(${test_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin"
    )
    target_compile_options(${test_target} PRIVATE -Wno-error)
    gtest_discover_tests(${test_target})
endforeach()
