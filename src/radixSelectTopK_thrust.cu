#include "radixSelectTopK_thrust.cuh"
#include "utils.cuh"

// 模板函数实现
template<typename KeyT>
cudaError_t radixSelectTopKThrust(
    KeyT *d_keys_in, 
    unsigned int num_items, 
    unsigned int k, 
    KeyT *d_keys_out, 
    unsigned int *d_indices_out, 
    cudaStream_t stream) {
  
  cudaError_t error = cudaSuccess;
  
  // 分配临时存储
  KeyT *d_keys_sorted = nullptr;
  unsigned int *d_indices_in = nullptr;
  
  cudaMalloc(&d_keys_sorted, sizeof(KeyT) * num_items);
  cudaMalloc(&d_indices_in, sizeof(unsigned int) * num_items);
  
  // 初始化索引数组
  unsigned int block_size = 256;
  unsigned int grid_size = (num_items + block_size - 1) / block_size;
  radixtopk::init_indices_kernel<<<grid_size, block_size, 0, stream>>>(d_indices_in, num_items);
  
  // 复制输入键到排序数组
  cudaMemcpyAsync(d_keys_sorted, d_keys_in, sizeof(KeyT) * num_items, 
                 cudaMemcpyDeviceToDevice, stream);
  
  // 使用Thrust进行排序（降序）
  thrust::sort_by_key(thrust::cuda::par.on(stream), 
                     d_keys_sorted, d_keys_sorted + num_items,
                     d_indices_in,
                     thrust::greater<KeyT>());
  
  // 复制Top-K结果
  cudaMemcpyAsync(d_keys_out, d_keys_sorted, sizeof(KeyT) * k, 
                 cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(d_indices_out, d_indices_in, sizeof(unsigned int) * k, 
                 cudaMemcpyDeviceToDevice, stream);
  
  // 释放临时内存
  cudaFree(d_keys_sorted);
  cudaFree(d_indices_in);
  
  return error;
}

template<typename KeyT>
cudaError_t radixSelectTopKPreallocatedThrust(
    const KeyT *d_keys_in, 
    unsigned int num_items, 
    unsigned int k, 
    KeyT *d_keys_out, 
    unsigned int *d_indices_out,
    KeyT *d_keys_sorted, 
    unsigned int *d_indices_in,
    cudaStream_t stream) {
  
  cudaError_t error = cudaSuccess;
  
  // 初始化索引数组
  unsigned int block_size = 256;
  unsigned int grid_size = (num_items + block_size - 1) / block_size;
  radixtopk::init_indices_kernel<<<grid_size, block_size, 0, stream>>>(d_indices_in, num_items);
  
  // 复制输入键到排序数组
  cudaMemcpyAsync(d_keys_sorted, d_keys_in, sizeof(KeyT) * num_items, 
                 cudaMemcpyDeviceToDevice, stream);
  
  // 使用Thrust进行排序（降序）
  thrust::sort_by_key(thrust::cuda::par.on(stream), 
                     d_keys_sorted, d_keys_sorted + num_items,
                     d_indices_in,
                     thrust::greater<KeyT>());
  
  // 复制Top-K结果
  cudaMemcpyAsync(d_keys_out, d_keys_sorted, sizeof(KeyT) * k, 
                 cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(d_indices_out, d_indices_in, sizeof(unsigned int) * k, 
                 cudaMemcpyDeviceToDevice, stream);
  
  return error;
}

// 显式实例化模板函数，确保编译器生成所需的代码
template cudaError_t radixSelectTopKThrust<float>(
    float*, unsigned int, unsigned int, 
    float*, unsigned int*, cudaStream_t);

template cudaError_t radixSelectTopKThrust<__half>(
    __half*, unsigned int, unsigned int, 
    __half*, unsigned int*, cudaStream_t);

template cudaError_t radixSelectTopKPreallocatedThrust<float>(
    const float*, unsigned int, unsigned int, 
    float*, unsigned int*, float*, unsigned int*, cudaStream_t);

template cudaError_t radixSelectTopKPreallocatedThrust<__half>(
    const __half*, unsigned int, unsigned int, 
    __half*, unsigned int*, __half*, unsigned int*, cudaStream_t);
