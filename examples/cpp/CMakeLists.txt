add_executable(quick_start quick_start.cpp)
target_link_libraries(quick_start PRIVATE rcl_core)

add_executable(mackey_glass mackey_glass.cpp)
target_link_libraries(mackey_glass PRIVATE rcl_core)

add_executable(online_learning online_learning.cpp)
target_link_libraries(online_learning PRIVATE rcl_core)
