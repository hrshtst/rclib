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
    ${Eigen3_INCLUDE_DIRS}
)

target_link_libraries(rcl_core PUBLIC
    Eigen3::Eigen
)
