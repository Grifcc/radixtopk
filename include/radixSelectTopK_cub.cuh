#ifndef RADIX_SELECT_TOP_K_CUB_CUH_
#define RADIX_SELECT_TOP_K_THRUST_CUH_

#include <cuda.h>
#include <cuda_fp16.h> // 添加half精度支持
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>
#include "utils.cuh"

// 原始版本
template <typename KeyT>
cudaError_t radixSelectTopKCub(KeyT *d_keys_in, unsigned int num_items, unsigned int k, KeyT *d_keys_out, unsigned int *d_indices_out,
                               cub::CachingDeviceAllocator &g_allocator, cudaStream_t stream = 0);

// 使用预分配数组的TopK实现 (只预分配排序结果数组)
template <typename KeyT>
cudaError_t radixSelectTopKPreallocatedCub(KeyT *d_keys_in,
                                           unsigned int num_items,
                                           unsigned int k,
                                           KeyT *d_keys_out,
                                           unsigned int *d_indices_out,
                                           KeyT *d_keys_sorted,
                                           unsigned int *d_indices_in,
                                           unsigned int *d_indices_sorted,
                                           cub::CachingDeviceAllocator &g_allocator,
                                           cudaStream_t stream = 0);

#endif // RADIX_SELECT_TOP_K_CUB_CUH_