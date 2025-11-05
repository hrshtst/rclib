pybind11_add_module(_rcl
    rcl/_rcl.cpp
)

target_link_libraries(_rcl PRIVATE rcl_core)
