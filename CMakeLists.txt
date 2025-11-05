cmake_minimum_required(VERSION 3.15)
project(rcl LANGUAGES CXX)

# Add subdirectories for third-party dependencies
add_subdirectory(cpp_core/third_party/catch2)
find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(cpp_core/third_party/eigen)

# Add the C++ core subdirectory
add_subdirectory(cpp_core)

# Add the Python binding subdirectory
add_subdirectory(python)
