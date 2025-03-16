add_library(layer
    layer.cpp
    layer.hpp
)

target_include_directories(layer PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(layer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(layer
    PRIVATE
    tensor
    buffer
    type
    status
    glog::glog
    CUDA::cudart
)