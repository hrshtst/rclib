pybind11_add_module(_rclib
    rclib/_rclib.cpp
)

target_link_libraries(_rclib PRIVATE rcl_core)

install(TARGETS _rclib DESTINATION rclib)