include(CTest)
include(Catch)

add_executable(test_reservoir test_reservoir.cpp)
target_link_libraries(test_reservoir PRIVATE rcl_core Catch2::Catch2WithMain)
catch_discover_tests(test_reservoir)

add_executable(test_nvar_reservoir test_nvar_reservoir.cpp)
target_link_libraries(test_nvar_reservoir PRIVATE rcl_core Catch2::Catch2WithMain)
catch_discover_tests(test_nvar_reservoir)

add_executable(test_readout test_readout.cpp)
target_link_libraries(test_readout PRIVATE rcl_core Catch2::Catch2WithMain)
catch_discover_tests(test_readout)

add_executable(test_rls_readout test_rls_readout.cpp)
target_link_libraries(test_rls_readout PRIVATE rcl_core Catch2::Catch2WithMain)
catch_discover_tests(test_rls_readout)
