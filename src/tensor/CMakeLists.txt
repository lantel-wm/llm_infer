add_library(tensor
    tensor.cpp
    tensor.hpp
)

target_include_directories(tensor PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(tensor
    PRIVATE
    buffer
    type
    glog::glog
    CUDA::cudart
)