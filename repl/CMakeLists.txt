add_executable(PolangRepl
    src/main.cpp
    src/repl_session.cpp
)

target_include_directories(PolangRepl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/compiler/include
    ${CMAKE_SOURCE_DIR}/parser/include
    ${LLVM_INCLUDE_DIRS}
)
target_compile_definitions(PolangRepl PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_directories(PolangRepl PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(PolangRepl PRIVATE PolangCodegen ${llvm_libs} LLVMAsmParser)
