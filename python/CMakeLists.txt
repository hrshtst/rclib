pybind11_add_module(_rclib rclib/_rclib.cpp)

target_link_libraries(_rclib PRIVATE rclib_core)

install(TARGETS _rclib DESTINATION rclib)
