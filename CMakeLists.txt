cmake_minimum_required(VERSION 3.10)
project(RadixTopK LANGUAGES CXX CUDA)

# 设置CUDA编译选项
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)  # 更新为C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 86)  # 针对AGX Orin (GA102)

# 设置CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# 导入CUDA库
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "    CUDA compiler: ${CUDA_NVCC_EXECUTABLE}")
set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# 启用调试信息（可选）
option(ENABLE_DEBUG "启用调试信息" OFF)
if(ENABLE_DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
else()
  set(CMAKE_BUILD_TYPE Release)
endif()

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# 添加测试可执行文件
add_executable(test_radixtopk_thrust
test_radix_thrust.cu
   src/utils.cu
   src/radixSelectTopK_thrust.cu
  
)

# 添加测试可执行文件
add_executable(test_radixtopk_cub
   test_radix_cub.cu
   src/utils.cu
   src/radixSelectTopK_cub.cu

)

# 链接库
target_link_libraries(test_radixtopk_thrust
  PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
)

# 链接库
target_link_libraries(test_radixtopk_cub
  PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
)

target_compile_options(test_radixtopk_cub PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --use_fast_math
>)

target_compile_options(test_radixtopk_thrust PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --use_fast_math
>)

