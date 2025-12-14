cmake_minimum_required(VERSION 3.20)

project(cieft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CIEFT_MCPU "apple-m1" CACHE STRING "AppleClang -mcpu value (e.g. apple-m1, apple-m2, native)")

add_executable(inspect
  src/gguf.cpp
  src/inspect.cpp
)

target_compile_options(inspect PRIVATE -Wall -Wextra -Wpedantic)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(NOT CIEFT_MCPU STREQUAL "")
    target_compile_options(inspect PRIVATE "-mcpu=${CIEFT_MCPU}")
  endif()
endif()

# Convenience: place binaries in the repo root so you can run `./inspect ...`
set_target_properties(inspect PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

