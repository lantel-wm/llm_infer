# Find GTest package
find_package(GTest REQUIRED)

# Add add kernel test executable
add_executable(test_add_kernel_cpu
    test_add_kernel_cpu.cpp
)

# Set include directories
target_include_directories(test_add_kernel_cpu
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/memory
    ${CUDAToolkit_INCLUDE_DIRS}
    ${glog_SOURCE_DIR}/src
    ${glog_BINARY_DIR}
)

# Link libraries
target_link_libraries(test_add_kernel_cpu
    PRIVATE
    GTest::GTest
    GTest::Main
    kernel
    tensor
    type
    add_kernel_cpu
    glog::glog
    buffer
) 

add_test(
    NAME test_add_kernel_cpu
    COMMAND test_add_kernel_cpu
)