cmake_minimum_required(VERSION 3.15)
project(rclib LANGUAGES CXX)

# Find OpenMP and set flags
find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  add_definitions(-DEIGEN_DONT_PARALLELIZE=0)
else()
  message(WARNING "OpenMP not found. The project will be built without parallelization.")
endif()
enable_testing()

# Add subdirectories for third-party dependencies
add_subdirectory(cpp_core/third_party/catch2)
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

option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()
