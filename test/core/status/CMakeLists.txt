# Find GTest package
find_package(GTest REQUIRED)

# Add status test executable
add_executable(test_status
    test_status.cpp
)

# Set include directories
target_include_directories(test_status
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(test_status
    PRIVATE
    status
    glog::glog
    GTest::GTest
    GTest::Main
)

# Add test to CTest
add_test(
    NAME test_status
    COMMAND test_status
) 