#ifndef RADIX_SELECT_TOP_K_THRUST_CUH_
#define RADIX_SELECT_TOP_K_THRUST_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 使用Thrust库替代CUB
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "utils.cuh"

/**
 * 使用Thrust的排序功能实现TopK
 * @param d_keys_in 输入键数组（设备内存）
 * @param num_items 输入数组长度
 * @param k 要返回的topK元素数量
 * @param d_keys_out 输出的topK键数组（设备内存）
 * @param d_indices_out 输出的topK索引数组（设备内存）
 * @param stream CUDA流
 * @return cudaError_t CUDA操作状态
 */
template<typename KeyT>
cudaError_t radixSelectTopKThrust(
    KeyT *d_keys_in, 
    unsigned int num_items, 
    unsigned int k, 
    KeyT *d_keys_out, 
    unsigned int *d_indices_out, 
    cudaStream_t stream = 0);

/**
 * 使用预分配内存的TopK实现
 * @param d_keys_in 输入键数组（设备内存）
 * @param num_items 输入数组长度
 * @param k 要返回的topK元素数量
 * @param d_keys_out 输出的topK键数组（设备内存）
 * @param d_indices_out 输出的topK索引数组（设备内存）
 * @param d_keys_sorted 预分配的排序键数组（设备内存）
 * @param d_indices_in 预分配的索引数组（设备内存）
 * @param stream CUDA流
 * @return cudaError_t CUDA操作状态
 */
template<typename KeyT>
cudaError_t radixSelectTopKPreallocatedThrust(
    const KeyT *d_keys_in, 
    unsigned int num_items, 
    unsigned int k, 
    KeyT *d_keys_out, 
    unsigned int *d_indices_out,
    KeyT *d_keys_sorted, 
    unsigned int *d_indices_in,
    cudaStream_t stream = 0);

#endif // RADIX_SELECT_TOP_K_THRUST_CUH_

