# Add status library
add_library(status
    status.hpp
    status.cpp
)

target_include_directories(status PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(status INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(status
    PRIVATE
    glog::glog
)