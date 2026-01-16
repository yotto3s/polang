add_executable(PolangCompiler
    src/main.cpp
    src/codegen.cpp
)

target_include_directories(PolangCompiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(PolangCompiler PRIVATE PolangParser)
