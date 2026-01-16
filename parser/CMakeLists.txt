find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

BISON_TARGET(PolangBison src/parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(PolangFlex src/lexer.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
ADD_FLEX_BISON_DEPENDENCY(PolangFlex PolangBison)

add_library(PolangParser STATIC
    ${BISON_PolangBison_OUTPUTS}
    ${FLEX_PolangFlex_OUTPUTS}
    src/parser_api.cpp
    src/node.cpp
    src/type_checker.cpp
    src/ast_printer.cpp
)

target_include_directories(PolangParser PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
)

target_compile_options(PolangParser PRIVATE -Wno-error)
