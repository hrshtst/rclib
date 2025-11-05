add_executable(quick_start quick_start.cpp)
target_link_libraries(quick_start PRIVATE rcl_core)

add_executable(mackey_glass mackey_glass.cpp)
target_link_libraries(mackey_glass PRIVATE rcl_core)
