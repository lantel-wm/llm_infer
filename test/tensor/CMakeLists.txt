# Find packages
find_package(GTest REQUIRED)

# Test tensor source
add_executable(test_tensor 
    test_tensor.cpp
)

# Include directories
target_include_directories(test_tensor PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(test_tensor PRIVATE
    GTest::GTest
    GTest::Main
    glog::glog
    tensor
    type
    buffer
    ${CUDA_LIBRARIES}
)

# Add test
add_test(
    NAME test_tensor 
    COMMAND test_tensor
)
