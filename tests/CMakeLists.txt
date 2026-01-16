# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
# Prevent overriding parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Lexer tests
add_executable(lexer_test
    lexer_test.cpp
)
target_link_libraries(lexer_test
    PRIVATE PolangParser
    PRIVATE PolangNodeStubs
    PRIVATE GTest::gtest_main
)
target_compile_options(lexer_test PRIVATE -Wno-error)

# Parser tests
add_executable(parser_test
    parser_test.cpp
)
target_link_libraries(parser_test
    PRIVATE PolangParser
    PRIVATE PolangNodeStubs
    PRIVATE GTest::gtest_main
)
target_compile_options(parser_test PRIVATE -Wno-error)

include(GoogleTest)
gtest_discover_tests(lexer_test)
gtest_discover_tests(parser_test)
