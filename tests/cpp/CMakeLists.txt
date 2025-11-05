include(CTest)
include(Catch)

add_executable(test_reservoir test_reservoir.cpp)
target_link_libraries(test_reservoir PRIVATE rcl_core Catch2::Catch2WithMain)

catch_discover_tests(test_reservoir)