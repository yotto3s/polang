# Fetch Google Test ONCE at root level
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Common test utilities directory
set(POLANG_TEST_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

# Include component test subdirectories
add_subdirectory(parser)
add_subdirectory(compiler)
add_subdirectory(repl)

# Include lit tests
add_subdirectory(lit)
