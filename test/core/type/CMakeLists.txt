# Find GTest package
find_package(GTest REQUIRED)

# Add status test executable
add_executable(test_type
    test_type.cpp
)

# Set include directories
target_include_directories(test_type
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(test_type
    PRIVATE
    type
    glog::glog
    GTest::GTest
    GTest::Main
)

# Add test to CTest
add_test(
    NAME test_type
    COMMAND test_type
) 