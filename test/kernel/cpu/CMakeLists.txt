# Find GTest package
find_package(GTest REQUIRED)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch"
    RESULT_VARIABLE PYTORCH_FOUND
    OUTPUT_QUIET
    ERROR_QUIET
)

message(STATUS "PYTHON3: ${Python3_EXECUTABLE}")

if(NOT ${PYTORCH_FOUND} EQUAL 0)
    message(WARNING "PyTorch not found. Some tests may be skipped.")
endif()

# Find all test files
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_*_kernel_cpu.cpp")

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
        ${CUDAToolkit_INCLUDE_DIRS}
        ${glog_SOURCE_DIR}/src
        ${glog_BINARY_DIR}
        ${Python3_INCLUDE_DIRS}
    )
    
    # Extract kernel name from test name (remove "test_" prefix)
    string(REGEX REPLACE "^test_" "" KERNEL_NAME ${TEST_NAME})
    
    # Link libraries
    target_link_libraries(${TEST_NAME}
        PRIVATE
        GTest::GTest
        GTest::Main
        kernel
        tensor
        type
        ${KERNEL_NAME}
        glog::glog
        buffer
        Python3::Python
    )
    
    # Add compile definition if PyTorch is not found
    if(NOT ${PYTORCH_FOUND} EQUAL 0)
        target_compile_definitions(${TEST_NAME} PRIVATE PYTORCH_NOT_FOUND)
    endif()
    
    # Add to test targets list
    list(APPEND TEST_TARGETS ${TEST_NAME})
    
    # Add the test to CTest
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${TEST_NAME}
    )
endforeach()