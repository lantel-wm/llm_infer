add_library(memory_manager
    memory_manager.cpp
    memory_manager.hpp
)

target_include_directories(memory_manager PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(memory_manager INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(memory_manager
    PRIVATE
    type
    glog::glog
    CUDA::cudart
)

# Add buffer library
add_library(buffer
    buffer.cpp
    buffer.hpp
)

target_include_directories(buffer PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(buffer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(buffer
    PRIVATE
    memory_manager
    type
    glog::glog
)