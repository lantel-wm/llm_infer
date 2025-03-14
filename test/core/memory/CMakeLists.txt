# Find GTest package
find_package(GTest REQUIRED)

# Add memory manager test executable
add_executable(test_memory_manager
    test_memory_manager.cpp
)

# Set include directories
target_include_directories(test_memory_manager
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(test_memory_manager
    PRIVATE
    memory_manager
    type
    glog::glog
    GTest::GTest
    GTest::Main
    CUDA::cudart
)

# Add test to CTest
add_test(
    NAME test_memory_manager
    COMMAND test_memory_manager
)

# Add buffer test executable
add_executable(test_buffer
    test_buffer.cpp
)

# Set include directories
target_include_directories(test_buffer
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(test_buffer
    PRIVATE
    buffer
    memory_manager
    type
    glog::glog
    GTest::GTest
    GTest::Main
    CUDA::cudart
)

# Add test to CTest
add_test(
    NAME test_buffer
    COMMAND test_buffer
)
