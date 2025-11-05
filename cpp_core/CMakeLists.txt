add_library(rcl_core STATIC
    src/Reservoir.cpp
)

target_include_directories(rcl_core PUBLIC
    include
    ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(rcl_core PUBLIC
    Eigen3::Eigen
)
