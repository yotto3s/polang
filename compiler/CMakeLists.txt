add_executable(PolangCompiler
    src/main.cpp
    src/codegen.cpp
)

target_include_directories(PolangCompiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
)

target_compile_definitions(PolangCompiler PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_libraries(PolangCompiler PRIVATE PolangParser)
