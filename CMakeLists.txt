cmake_minimum_required(VERSION 3.20)
project(Polang LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Werror)

# Code coverage option
option(POLANG_ENABLE_COVERAGE "Enable code coverage with gcov" OFF)

if(POLANG_ENABLE_COVERAGE)
  message(STATUS "Code coverage enabled")
  add_compile_options(--coverage)
  add_link_options(--coverage)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find MLIR (which includes LLVM)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Store LLVM/MLIR include directories for use by targets that need them
# We DON'T add these globally to avoid LLVM's cxxabi.h conflicting with
# system headers when building GoogleTest
set(POLANG_LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
set(POLANG_MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS})

# These project directories can be added globally (no conflicts)
include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/mlir/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Get LLVM libraries (still needed for JIT)
llvm_map_components_to_libnames(llvm_libs core support native OrcJIT)

add_subdirectory(parser)
add_subdirectory(mlir)
add_subdirectory(compiler)
add_subdirectory(repl)

# Enable testing
enable_testing()
add_subdirectory(tests)

# Coverage report targets
if(POLANG_ENABLE_COVERAGE)
  find_program(LCOV_PATH lcov)
  find_program(GENHTML_PATH genhtml)

  if(LCOV_PATH AND GENHTML_PATH)
    add_custom_target(coverage
      COMMAND ${LCOV_PATH} --capture --directory ${CMAKE_BINARY_DIR}
              --output-file coverage.info
              --exclude '/usr/*'
              --exclude '${CMAKE_BINARY_DIR}/*'
              --exclude '*/tests/*'
              --ignore-errors gcov
      COMMAND ${GENHTML_PATH} coverage.info --output-directory coverage_html
              --ignore-errors unmapped
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating HTML coverage report in build/coverage_html/"
    )

    add_custom_target(coverage-clean
      COMMAND ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --zerocounters
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Resetting coverage counters"
    )
  else()
    message(WARNING "lcov/genhtml not found - coverage targets disabled")
  endif()
endif()
