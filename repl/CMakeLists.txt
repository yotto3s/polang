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
# Export symbols so JIT-compiled code can resolve runtime library functions
# (e.g., __polang_runtime_error). We force the linker to keep all symbols from
# PolangRuntime even though nothing in PolangRepl calls them directly.
# ENABLE_EXPORTS (-rdynamic) makes them visible to dlsym/JIT.
set_target_properties(PolangRepl PROPERTIES ENABLE_EXPORTS TRUE)
if(APPLE)
  target_link_libraries(PolangRepl PRIVATE
      PolangReplLib
      -Wl,-all_load PolangRuntime
  )
else()
  target_link_libraries(PolangRepl PRIVATE
      PolangReplLib
      -Wl,--whole-archive PolangRuntime -Wl,--no-whole-archive
  )
endif()
