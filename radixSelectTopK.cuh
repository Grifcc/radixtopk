#include <cuda.h>
#include <cuda_fp16.h> // 添加half精度支持
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>

using namespace cub;
using namespace std;

/**
 * 初始化索引数组的简单内核
 */
__global__ void init_indices_kernel(uint* indices, uint num_items) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    indices[idx] = idx;
  }
}

/**
 * 使用CUB的排序功能实现TopK
 */
template<typename KeyT>
cudaError_t radixSelectTopK(KeyT *d_keys_in, uint num_items, uint k, KeyT *d_keys_out, uint *d_indices_out,
    CachingDeviceAllocator &g_allocator) {
  
  cudaError_t error = cudaSuccess;
  
  // 临时存储空间
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  
  // 初始化索引数组
  uint *d_indices_in = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_indices_in, sizeof(uint) * num_items));
  
  // 初始化索引数组为[0,1,2,...,num_items-1]
  uint block_size = 256;
  uint grid_size = (num_items + block_size - 1) / block_size;
  init_indices_kernel<<<grid_size, block_size>>>(d_indices_in, num_items);
  CubDebugExit(cudaGetLastError());
  
  // 为排序设置临时键和索引数组
  KeyT *d_keys_sorted = NULL;
  uint *d_indices_sorted = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_sorted, sizeof(KeyT) * num_items));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_indices_sorted, sizeof(uint) * num_items));
  
  // 计算排序所需的临时存储空间大小
  CubDebugExit(DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_sorted,
        d_indices_in, d_indices_sorted,
        num_items));
  
  // 分配临时存储
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  
  // 执行排序
  CubDebugExit(DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_sorted,
        d_indices_in, d_indices_sorted,
        num_items));
  
  // 复制排序结果的前k个元素到输出
  CubDebugExit(cudaMemcpy(d_keys_out, d_keys_sorted, k * sizeof(KeyT), cudaMemcpyDeviceToDevice));
  CubDebugExit(cudaMemcpy(d_indices_out, d_indices_sorted, k * sizeof(uint), cudaMemcpyDeviceToDevice));
  
  // 释放临时内存
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  if (d_keys_sorted) CubDebugExit(g_allocator.DeviceFree(d_keys_sorted));
  if (d_indices_sorted) CubDebugExit(g_allocator.DeviceFree(d_indices_sorted));
  if (d_indices_in) CubDebugExit(g_allocator.DeviceFree(d_indices_in));
  
  return error;
}

