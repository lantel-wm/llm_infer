cmake_minimum_required(VERSION 3.18)
project(llm_infer LANGUAGES CUDA CXX)

# Ensure compile commands are exported with full command line
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_NINJA_FORCE_RESPONSE_FILE OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

# Set CUDA debug flags
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -lineinfo")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


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

# Find LAPACK and BLAS
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# Fetch and make Armadillo available
FetchContent_Declare(
    armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG 14.0.2
)
set(BUILD_SHARED_LIBS OFF)  # Build static libraries
set(ARMA_USE_LAPACK ON)     # Enable LAPACK support
set(ARMA_USE_BLAS ON)       # Enable BLAS support
set(ARMA_NO_DEBUG ON)       # Disable debug mode for better performance
FetchContent_MakeAvailable(armadillo)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test) 