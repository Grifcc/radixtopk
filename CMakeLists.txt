cmake_minimum_required(VERSION 3.10)
project(topk LANGUAGES CXX CUDA)

# 设置CUDA编译选项
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)  # 更新为C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES 86)  # 针对AGX Orin (GA102)

# 查找CUDA包
# CUDA Configuration
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "    CUDA compiler: ${CUDA_NVCC_EXECUTABLE}")
set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})

# 添加头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# 创建基准测试可执行文件
add_executable(test_radix
   test_radix.cu radixSelectTopK.cuh
)

# 链接库
target_link_libraries(test_radix
    CUDA::cudart
    CUDA::cuda_driver
)