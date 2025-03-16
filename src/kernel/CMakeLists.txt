add_library(kernel
    kernel.cpp
    kernel.hpp
)

target_include_directories(kernel PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(kernel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(kernel
    PRIVATE
    tensor
    buffer
    type
    config
    glog::glog
    CUDA::cudart
)

# Add CPU kernels subdirectory
add_subdirectory(cpu)
add_subdirectory(gpu)

# Link CPU kernel libraries to the main kernel library
target_link_libraries(kernel PRIVATE ${CPU_KERNEL_LIBS})
target_link_libraries(kernel PRIVATE ${GPU_KERNEL_LIBS})