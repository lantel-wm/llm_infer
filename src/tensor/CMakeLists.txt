add_library(tensor
    tensor.cpp
    tensor.hpp
)

# Make headers available for both regular C++ and CUDA
target_include_directories(tensor 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tensor
    PUBLIC
        buffer
        type
        CUDA::cudart
    PRIVATE
        glog::glog
)