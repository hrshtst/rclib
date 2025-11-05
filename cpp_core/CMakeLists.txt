add_library(rcl_core STATIC
    src/Reservoir.cpp
    src/Model.cpp
    src/reservoirs/RandomSparseReservoir.cpp
    src/reservoirs/NvarReservoir.cpp
    src/readouts/RidgeReadout.cpp
    src/readouts/RlsReadout.cpp
    src/readouts/LmsReadout.cpp
)

target_include_directories(rcl_core PUBLIC
    include
    ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(rcl_core PUBLIC
    Eigen3::Eigen
)