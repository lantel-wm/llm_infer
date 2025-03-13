cmake_minimum_required(VERSION 3.18)
project(llm_infer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure we use the system C++ standard library
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")

# Include FetchContent
include(FetchContent)

# Fetch and make gflags available
FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG v2.2.2
)
set(BUILD_SHARED_LIBS OFF)  # Build static libraries
set(INSTALL_HEADERS OFF)    # Don't install headers
set(BUILD_TESTING OFF)      # Don't build tests
FetchContent_MakeAvailable(gflags)

# Fetch and make glog available
FetchContent_Declare(
    glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG v0.7.1
)
set(BUILD_SHARED_LIBS OFF)  # Build static libraries
set(INSTALL_HEADERS OFF)    # Don't install headers
set(BUILD_TESTING OFF)      # Don't build tests
set(WITH_GFLAGS OFF)        # Don't use system gflags
FetchContent_MakeAvailable(glog)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src/core)
add_subdirectory(test) 