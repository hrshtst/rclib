# Find OpenMP
find_package(OpenMP)

add_library(rcl_core STATIC
    src/Reservoir.cpp
    src/Readout.cpp
    src/Model.cpp
    src/reservoirs/RandomSparseReservoir.cpp
    src/reservoirs/NvarReservoir.cpp
    src/readouts/RidgeReadout.cpp
    src/readouts/RlsReadout.cpp
    src/readouts/LmsReadout.cpp
)

set_target_properties(rcl_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(rcl_core PUBLIC
    include
    ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(rcl_core PUBLIC
    Eigen3::Eigen
)

if(OpenMP_FOUND)
    target_link_libraries(rcl_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(rcl_core PUBLIC EIGEN_DONT_PARALLELIZE=0)
else()
    message(WARNING "OpenMP not found. The project will be built without parallelization.")
endif()
