cmake_minimum_required(VERSION 3.15)
project(rclib LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()

enable_testing()

# Add subdirectories for third-party dependencies
add_subdirectory(cpp_core/third_party/catch2)
set(PYBIND11_FINDPYTHON NEW)
add_subdirectory(cpp_core/third_party/pybind11)
add_subdirectory(cpp_core/third_party/eigen)

# Add the C++ core subdirectory
add_subdirectory(cpp_core)

# Add the Python binding subdirectory
add_subdirectory(python)

# Add the C++ tests subdirectory
if(BUILD_TESTING)
  add_subdirectory(tests/cpp)
endif()

option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES)
  add_subdirectory(examples/cpp)
endif()
