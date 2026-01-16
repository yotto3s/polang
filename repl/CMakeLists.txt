# REPL session library (for testing)
add_library(PolangReplLib STATIC
    src/repl_session.cpp
)

target_include_directories(PolangReplLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/compiler/include
    ${CMAKE_SOURCE_DIR}/parser/include
    ${CMAKE_SOURCE_DIR}/mlir/include
    ${CMAKE_BINARY_DIR}/mlir/include
    ${POLANG_LLVM_INCLUDE_DIRS}
    ${POLANG_MLIR_INCLUDE_DIRS}
)
target_compile_definitions(PolangReplLib PUBLIC ${LLVM_DEFINITIONS_LIST})
target_link_directories(PolangReplLib PUBLIC ${LLVM_LIBRARY_DIRS})
target_link_libraries(PolangReplLib PUBLIC PolangMLIRCodegen)

# REPL executable
add_executable(PolangRepl src/main.cpp)

target_include_directories(PolangRepl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(PolangRepl PRIVATE PolangReplLib)
