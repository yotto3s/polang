add_executable(PolangRepl src/main.cpp)

target_include_directories(PolangRepl PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(PolangRepl PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_directories(PolangRepl PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(PolangRepl PRIVATE ${llvm_libs} LLVMAsmParser)
