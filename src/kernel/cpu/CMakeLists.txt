# Find all cpu kernel files
file(GLOB CPU_KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel_cpu.hpp")
file(GLOB CPU_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel_cpu.cpp")

# Create a list to store all CPU kernel libraries
set(CPU_KERNEL_LIBS)

# Create a library for each kernel
foreach(KERNEL_SOURCE ${CPU_KERNEL_SOURCES})
    # Extract the kernel name from the file path
    get_filename_component(KERNEL_NAME ${KERNEL_SOURCE} NAME_WE)
    
    # Find the corresponding header file
    set(KERNEL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_NAME}.hpp")
    
    # Create the library
    add_library(${KERNEL_NAME}
        ${KERNEL_SOURCE}
        ${KERNEL_HEADER}
    )
    
    # Set include directories
    target_include_directories(${KERNEL_NAME} 
        PRIVATE 
        ${CMAKE_SOURCE_DIR}/src/tensor
        ${CMAKE_SOURCE_DIR}/src/core/memory
        ${CMAKE_SOURCE_DIR}/src/core/type
        ${CMAKE_SOURCE_DIR}/src/core/config
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_include_directories(${KERNEL_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    
    # Link with necessary dependencies
    target_link_libraries(${KERNEL_NAME}
        PRIVATE
        tensor
        buffer
        type
        config
        armadillo
        glog::glog
        CUDA::cudart
    )
    
    # Add to the list of CPU kernel libraries
    list(APPEND CPU_KERNEL_LIBS ${KERNEL_NAME})
endforeach()

# Export the list of CPU kernel libraries to parent scope
set(CPU_KERNEL_LIBS ${CPU_KERNEL_LIBS} PARENT_SCOPE)
