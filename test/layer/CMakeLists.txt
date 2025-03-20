# Find GTest package
find_package(GTest REQUIRED)

# Find all test files
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_*_layer.cpp")

# Create a list to store all test targets
set(TEST_TARGETS)

# Create test executables for each test file
foreach(TEST_SOURCE ${TEST_SOURCES})
    # Extract the test name from the file path
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Create the test executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Set include directories
    target_include_directories(${TEST_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core
        ${CMAKE_SOURCE_DIR}/src/core/memory
        ${CMAKE_SOURCE_DIR}/src/core/config
        ${CMAKE_SOURCE_DIR}/src/core/status
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    # Extract layer name from test name (remove "test_" prefix)
    # Example:test_vec_add_layer -> vec_add_layer
    string(REGEX REPLACE "^test_" "" LAYER_NAME ${TEST_NAME})
    
    # Link libraries
    target_link_libraries(${TEST_NAME}
        PRIVATE
        GTest::GTest
        GTest::Main
        layer
        ${LAYER_NAME}
        tensor
        type
        buffer
        config
        memory_manager
        status
        glog::glog
        CUDA::cudart
    )
    
    # Add to test targets list
    list(APPEND TEST_TARGETS ${TEST_NAME})
    
    # Add the test to CTest
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${TEST_NAME}
    )
endforeach()

message(STATUS "Layer test targets: ${TEST_TARGETS}")
