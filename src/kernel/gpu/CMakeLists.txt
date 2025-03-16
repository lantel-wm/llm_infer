# Find all gpu kernel files
file(GLOB GPU_KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel_gpu.cuh")
file(GLOB GPU_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel_gpu.cu")

# Enable CUDA
enable_language(CUDA)
# set(CMAKE_CUDA_ARCHITECTURES 52)

# Create a list to store all GPU kernel libraries
set(GPU_KERNEL_LIBS)

# Create a library for each kernel
foreach(KERNEL_SOURCE ${GPU_KERNEL_SOURCES})
    # Extract the kernel name from the file path
    get_filename_component(KERNEL_NAME ${KERNEL_SOURCE} NAME_WE)

    # Find the corresponding header file
    set(KERNEL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_NAME}.cuh")
    
    # Create the library
    add_library(${KERNEL_NAME} STATIC
        ${KERNEL_SOURCE}
    )

    set_target_properties(${KERNEL_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Set include directories
    target_include_directories(${KERNEL_NAME} 
        PRIVATE 
        ${CMAKE_SOURCE_DIR}/src/tensor
        ${CMAKE_SOURCE_DIR}/src/core/memory
        ${CMAKE_SOURCE_DIR}/src/core/type
        ${CMAKE_SOURCE_DIR}/src/core/config
        ${CMAKE_SOURCE_DIR}/build/_deps/glog-build
        ${CMAKE_SOURCE_DIR}/build/_deps/glog-src/src
        INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR} 
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    # Link with necessary dependencies
    target_link_libraries(${KERNEL_NAME}
        PRIVATE
        tensor
        buffer
        type
        config
        glog::glog
        CUDA::cudart
    )
    
    # Add to the list of GPU kernel libraries
    list(APPEND GPU_KERNEL_LIBS ${KERNEL_NAME})
endforeach()

message(STATUS "GPU_KERNEL_LIBS: ${GPU_KERNEL_LIBS}")

# Export the list of GPU kernel libraries to parent scope
set(GPU_KERNEL_LIBS ${GPU_KERNEL_LIBS} PARENT_SCOPE)
