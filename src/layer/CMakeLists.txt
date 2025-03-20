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
    config
    status
    glog::glog
    CUDA::cudart
)

# Find all layer files
file(GLOB LAYER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*_layer.cpp")

# Create a list to store all layer targets
set(LAYER_TARGETS)

# Create layer targets for each layer file
foreach(LAYER_SOURCE ${LAYER_SOURCES})
    # Extract the layer name from the file path
    get_filename_component(LAYER_NAME ${LAYER_SOURCE} NAME_WE)

    # Create a target for the layer
    add_library(${LAYER_NAME} STATIC ${LAYER_SOURCE})

    # Add the layer target to the list
    list(APPEND LAYER_TARGETS ${LAYER_NAME})

    # Set include directories
    target_include_directories(${LAYER_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/kernel
    ${CUDAToolkit_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR})

    # Link libraries
    target_link_libraries(${LAYER_NAME} PRIVATE
        tensor
        buffer
        type
        config
        status
        kernel
        glog::glog
        CUDA::cudart
    )

    # Add the layer target to the list
    list(APPEND LAYER_TARGETS ${LAYER_NAME})
endforeach()

# # Add vec_add library
# add_library(vec_add
#     vec_add.cpp
#     vec_add.hpp
# )

# target_include_directories(vec_add PRIVATE 
#     ${CUDAToolkit_INCLUDE_DIRS} 
#     ${CMAKE_SOURCE_DIR}/src/kernel
# )
# target_include_directories(vec_add INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(vec_add
#     PRIVATE
#     tensor
#     buffer
#     type
#     config
#     status
#     kernel
#     glog::glog
#     CUDA::cudart
# )