# Options
option(RCLIB_USE_OPENMP "Enable OpenMP support" ON)
option(RCLIB_ENABLE_EIGEN_PARALLELIZATION "Enable Eigen's internal parallelization" OFF)

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

    # Enable user-level parallelization only if Eigen's internal parallelization is disabled
    if(NOT RCLIB_ENABLE_EIGEN_PARALLELIZATION)
      target_compile_definitions(rclib_core PUBLIC RCLIB_ENABLE_USER_PARALLELIZATION)
    endif()
  else()
    message(WARNING "OpenMP not found. The project will be built without parallelization.")
  endif()
endif()

if(NOT RCLIB_ENABLE_EIGEN_PARALLELIZATION)
  target_compile_definitions(rclib_core PUBLIC EIGEN_DONT_PARALLELIZE)
endif()
