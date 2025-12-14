cmake_minimum_required(VERSION 3.20)

project(cieft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CIEFT_MCPU "apple-m1" CACHE STRING "AppleClang -mcpu value (e.g. apple-m1, apple-m2, native)")

add_library(cieft_core
  src/dequant_q4_k.cpp
  src/dequant_q6_k.cpp
  src/gguf.cpp
  src/gguf_loader.cpp
  src/layer0.cpp
  src/weights.cpp
)

target_compile_options(cieft_core PRIVATE -Wall -Wextra -Wpedantic)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(NOT CIEFT_MCPU STREQUAL "")
    target_compile_options(cieft_core PRIVATE "-mcpu=${CIEFT_MCPU}")
  endif()
endif()

add_executable(inspect src/inspect.cpp)
target_link_libraries(inspect PRIVATE cieft_core)

add_executable(smoke_load src/smoke_load.cpp)
target_link_libraries(smoke_load PRIVATE cieft_core)

add_executable(layer0_step src/layer0_step.cpp)
target_link_libraries(layer0_step PRIVATE cieft_core)

add_executable(two_layer_nn exercises/two_layer_nn.cpp)
target_compile_options(two_layer_nn PRIVATE -Wall -Wextra -Wpedantic)
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(NOT CIEFT_MCPU STREQUAL "")
    target_compile_options(two_layer_nn PRIVATE "-mcpu=${CIEFT_MCPU}")
  endif()
endif()

# Convenience: place binaries in the repo root so you can run `./inspect ...`
set_target_properties(inspect smoke_load layer0_step two_layer_nn PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
