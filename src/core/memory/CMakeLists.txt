add_library(memory_manager
    memory_manager.cpp
    memory_manager.hpp
)

target_include_directories(memory_manager PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(memory_manager
    PRIVATE
    type
    glog::glog
    CUDA::cudart
) 