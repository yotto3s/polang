# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# List of test names (each corresponds to <name>_test.cpp)
set(PARSER_TESTS
    lexer
    parser_declaration
    parser_expression
    parser_control_flow
    parser_statement
    error
    type_check
)

include(GoogleTest)

foreach(test_name ${PARSER_TESTS})
    set(test_target ${test_name}_test)
    add_executable(${test_target} ${test_target}.cpp)
    target_link_libraries(${test_target}
        PRIVATE PolangParser
        PRIVATE GTest::gtest_main
    )
    target_compile_options(${test_target} PRIVATE -Wno-error)
    gtest_discover_tests(${test_target})
endforeach()
