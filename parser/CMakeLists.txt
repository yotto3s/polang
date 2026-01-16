find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

BISON_TARGET(PolangBison src/parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(PolangFlex src/lexer.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
ADD_FLEX_BISON_DEPENDENCY(PolangFlex PolangBison)

add_library(PolangParser STATIC
    ${BISON_PolangBison_OUTPUTS}
    ${FLEX_PolangFlex_OUTPUTS}
    src/parser_api.cpp
)

# Stub library for testing (provides default codeGen implementations)
add_library(PolangNodeStubs STATIC src/node_stubs.cpp)
target_include_directories(PolangNodeStubs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
)

target_include_directories(PolangParser PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
)

# LLVM includes are PRIVATE since node.hpp uses forward declarations
target_include_directories(PolangParser PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(PolangParser PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_directories(PolangParser PUBLIC ${LLVM_LIBRARY_DIRS})
target_link_libraries(PolangParser PUBLIC ${llvm_libs})
target_compile_options(PolangParser PRIVATE -Wno-error)
