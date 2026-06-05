# Options
option(RCLIB_USE_OPENMP "Enable OpenMP support" ON)
option(RCLIB_ENABLE_EIGEN_PARALLELIZATION "Enable Eigen's internal parallelization" ON)
option(RCLIB_ADAPTIVE_PARALLELIZATION "Enable adaptive parallelization based on problem size" ON)

# Find OpenMP
if(RCLIB_USE_OPENMP)
  find_package(OpenMP)
endif()

add_library(
  rclib_core STATIC
  src/Reservoir.cpp
  src/Readout.cpp
  src/Model.cpp
  src/reservoirs/RandomSparseReservoir.cpp
  src/reservoirs/NvarReservoir.cpp
  src/readouts/RidgeReadout.cpp
  src/readouts/RlsReadout.cpp
  src/readouts/LmsReadout.cpp)

set_target_properties(rclib_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(rclib_core PUBLIC include ${EIGEN_INCLUDE_DIRS})

target_link_libraries(rclib_core PUBLIC Eigen3::Eigen)

if(RCLIB_USE_OPENMP)
  if(OpenMP_FOUND)
    target_link_libraries(rclib_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(rclib_core PUBLIC RCLIB_USE_OPENMP)
    if(RCLIB_ADAPTIVE_PARALLELIZATION)
      target_compile_definitions(rclib_core PUBLIC RCLIB_ADAPTIVE_PARALLELIZATION)
    endif()
  else()
    message(WARNING "OpenMP not found. The project will be built without parallelization.")
  endif()
endif()

if(NOT RCLIB_ENABLE_EIGEN_PARALLELIZATION)
  target_compile_definitions(rclib_core PUBLIC EIGEN_DONT_PARALLELIZE)
endif()

# Warnings on our own sources only (PRIVATE: not propagated to consumers or to third-party headers). RCLIB_WERROR
# promotes them to errors and is enabled in CI so the build stays clean.
option(RCLIB_WERROR "Treat compiler warnings as errors (used in CI)" OFF)
target_compile_options(rclib_core PRIVATE -Wall -Wextra -Wpedantic)
if(RCLIB_WERROR)
  target_compile_options(rclib_core PRIVATE -Werror)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Eigen's headers trip a known GCC -Wmaybe-uninitialized false positive at -O3; keep that one non-fatal so -Werror
    # still guards our own code.
    target_compile_options(rclib_core PRIVATE -Wno-error=maybe-uninitialized)
  endif()
endif()
