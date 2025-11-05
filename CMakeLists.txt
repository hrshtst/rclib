cmake_minimum_required(VERSION 3.15)
project(rcl LANGUAGES CXX)
enable_testing()

# Add subdirectories for third-party dependencies
add_subdirectory(cpp_core/third_party/catch2)
add_subdirectory(cpp_core/third_party/pybind11)
add_subdirectory(cpp_core/third_party/eigen)

# Add the C++ core subdirectory
add_subdirectory(cpp_core)

# Add the Python binding subdirectory
add_subdirectory(src)

# Add the C++ tests subdirectory
if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()
