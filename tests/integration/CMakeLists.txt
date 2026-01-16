# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# List of integration test names (each corresponds to <name>_test.cpp)
set(INTEGRATION_TESTS
    compiler
    repl
)

include(GoogleTest)

foreach(test_name ${INTEGRATION_TESTS})
    set(test_target ${test_name}_test)
    add_executable(${test_target} ${test_target}.cpp)
    target_link_libraries(${test_target}
        PRIVATE GTest::gtest_main
        PRIVATE GTest::gmock
    )
    set_target_properties(${test_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin"
    )
    target_compile_definitions(${test_target} PRIVATE
        POLANG_COMPILER_PATH="$<TARGET_FILE:PolangCompiler>"
        POLANG_REPL_PATH="$<TARGET_FILE:PolangRepl>"
    )
    target_compile_options(${test_target} PRIVATE -Wno-error)
    gtest_discover_tests(${test_target})
endforeach()
