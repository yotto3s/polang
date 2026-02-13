add_library(PolangRuntime STATIC src/polang_runtime.c)
target_include_directories(PolangRuntime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
