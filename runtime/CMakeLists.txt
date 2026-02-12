add_library(PolangRuntime STATIC
  src/runtime.c
)

target_include_directories(PolangRuntime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
