# Codegen library (shared between compiler and REPL)
# LLVM IR backend
add_library(PolangCodegen STATIC
    src/codegen.cpp
    src/codegen_visitor.cpp
)

target_include_directories(PolangCodegen PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
)

target_compile_definitions(PolangCodegen PUBLIC ${LLVM_DEFINITIONS_LIST})
target_link_directories(PolangCodegen PUBLIC ${LLVM_LIBRARY_DIRS})
target_link_libraries(PolangCodegen PUBLIC PolangParser ${llvm_libs})

# MLIR backend codegen library
add_library(PolangMLIRCodegen STATIC
    src/mlir_codegen.cpp
)

target_include_directories(PolangMLIRCodegen PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/mlir/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(PolangMLIRCodegen PUBLIC ${LLVM_DEFINITIONS_LIST})
target_link_directories(PolangMLIRCodegen PUBLIC ${LLVM_LIBRARY_DIRS})

# Suppress overloaded-virtual warning for MLIR code
target_compile_options(PolangMLIRCodegen PRIVATE -Wno-overloaded-virtual)

target_link_libraries(PolangMLIRCodegen PUBLIC
    PolangParser
    PolangMLIR
    MLIRExecutionEngine
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRPass
    MLIRTransforms
    ${llvm_libs}
)

# Compiler executable
add_executable(PolangCompiler src/main.cpp)

target_include_directories(PolangCompiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/mlir/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(PolangCompiler PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_directories(PolangCompiler PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(PolangCompiler PRIVATE PolangCodegen PolangMLIRCodegen)
