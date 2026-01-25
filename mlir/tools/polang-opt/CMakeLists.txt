set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(polang-opt
  polang-opt.cpp
)

target_link_libraries(polang-opt
  PRIVATE
  PolangDialect
  PolangASTDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIROptLib
)

target_include_directories(polang-opt PRIVATE
  ${PROJECT_SOURCE_DIR}/mlir/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../include
)
