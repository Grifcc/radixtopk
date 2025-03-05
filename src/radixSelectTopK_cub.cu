#include "radixSelectTopK_cub.cuh"

/**
 * 使用CUB的排序功能实现TopK
 */
template <typename KeyT>
cudaError_t radixSelectTopKCub(KeyT *d_keys_in, unsigned int num_items, unsigned int k, KeyT *d_keys_out, unsigned int *d_indices_out,
                               cub::CachingDeviceAllocator &g_allocator, cudaStream_t stream)
{

    cudaError_t error = cudaSuccess;

    // 临时存储空间
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // 初始化索引数组
    unsigned int *d_indices_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void **)&d_indices_in, sizeof(unsigned int) * num_items));

    // 初始化索引数组为[0,1,2,...,num_items-1]
    unsigned int block_size = 256;
    unsigned int grid_size = (num_items + block_size - 1) / block_size;
    radixtopk::init_indices_kernel<<<grid_size, block_size, 0, stream>>>(d_indices_in, num_items);
    CubDebugExit(cudaGetLastError());

    // 为排序设置临时键和索引数组
    KeyT *d_keys_sorted = NULL;
    unsigned int *d_indices_sorted = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void **)&d_keys_sorted, sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void **)&d_indices_sorted, sizeof(unsigned int) * num_items));

    // 计算排序所需的临时存储空间大小
    CubDebugExit(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_sorted,
        d_indices_in, d_indices_sorted,
        num_items));

    // 分配临时存储
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // 执行排序
    CubDebugExit(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_sorted,
        d_indices_in, d_indices_sorted,
        num_items));

    // 复制排序结果的前k个元素到输出
    CubDebugExit(cudaMemcpy(d_keys_out, d_keys_sorted, k * sizeof(KeyT), cudaMemcpyDeviceToDevice));
    CubDebugExit(cudaMemcpy(d_indices_out, d_indices_sorted, k * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    // 释放临时内存
    if (d_temp_storage)
        CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    if (d_keys_sorted)
        CubDebugExit(g_allocator.DeviceFree(d_keys_sorted));
    if (d_indices_sorted)
        CubDebugExit(g_allocator.DeviceFree(d_indices_sorted));
    if (d_indices_in)
        CubDebugExit(g_allocator.DeviceFree(d_indices_in));

    return error;
}

/**
 * 使用CUB的排序功能实现TopK (预分配排序结果数组)
 */
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
                                           cudaStream_t stream)
{

    cudaError_t error = cudaSuccess;

    // 临时存储空间
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // 初始化索引数组
    // unsigned int *d_indices_in = NULL;
    // CubDebugExit(g_allocator.DeviceAllocate((void **)&d_indices_in, sizeof(unsigned int) * num_items));

    // 初始化索引数组为[0,1,2,...,num_items-1]
    unsigned int block_size = 256;
    unsigned int grid_size = (num_items + block_size - 1) / block_size;
    radixtopk::init_indices_kernel<<<grid_size, block_size, 0, stream>>>(d_indices_in, num_items);
    CubDebugExit(cudaGetLastError());

    // // 为排序设置临时键和索引数组
    // KeyT *d_keys_sorted = NULL;
    // unsigned int *d_indices_sorted = NULL;
    // CubDebugExit(g_allocator.DeviceAllocate((void **)&d_keys_sorted, sizeof(KeyT) * num_items));
    // CubDebugExit(g_allocator.DeviceAllocate((void **)&d_indices_sorted, sizeof(unsigned int) * num_items));

    // 计算排序所需的临时存储空间大小
    CubDebugExit(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_sorted,
        d_indices_in, d_indices_sorted,
        num_items));

    // 分配临时存储
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // 执行排序
    CubDebugExit(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_sorted,
        d_indices_in, d_indices_sorted,
        num_items));

    // 复制排序结果的前k个元素到输出
    CubDebugExit(cudaMemcpy(d_keys_out, d_keys_sorted, k * sizeof(KeyT), cudaMemcpyDeviceToDevice));
    CubDebugExit(cudaMemcpy(d_indices_out, d_indices_sorted, k * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    // 释放临时内存
    if (d_temp_storage)
        CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    // if (d_keys_sorted)
    //     CubDebugExit(g_allocator.DeviceFree(d_keys_sorted));
    // if (d_indices_sorted)
    //     CubDebugExit(g_allocator.DeviceFree(d_indices_sorted));
    // if (d_indices_in)
    //     CubDebugExit(g_allocator.DeviceFree(d_indices_in));

    return error;
}

// 显式实例化模板函数，确保编译器生成所需的代码
template cudaError_t radixSelectTopKCub<float>(
    float *d_keys_in, unsigned int num_items, unsigned int k, float *d_keys_out, unsigned int *d_indices_out,
    cub::CachingDeviceAllocator &g_allocator, cudaStream_t stream);

template cudaError_t radixSelectTopKCub<__half>(
    __half *d_keys_in, unsigned int num_items, unsigned int k, __half *d_keys_out, unsigned int *d_indices_out,
    cub::CachingDeviceAllocator &g_allocator, cudaStream_t stream);

template cudaError_t radixSelectTopKPreallocatedCub<float>(float *d_keys_in,
                                                           unsigned int num_items,
                                                           unsigned int k,
                                                           float *d_keys_out,
                                                           unsigned int *d_indices_out,
                                                           float *d_keys_sorted,
                                                           unsigned int *d_indices_in,
                                                           unsigned int *d_indices_sorted,
                                                           cub::CachingDeviceAllocator &g_allocator,
                                                           cudaStream_t stream);
                                                           
template cudaError_t radixSelectTopKPreallocatedCub<__half>(__half *d_keys_in,
                                                            unsigned int num_items,
                                                            unsigned int k,
                                                            __half *d_keys_out,
                                                            unsigned int *d_indices_out,
                                                            __half *d_keys_sorted,
                                                            unsigned int *d_indices_in,
                                                            unsigned int *d_indices_sorted,
                                                            cub::CachingDeviceAllocator &g_allocator,
                                                            cudaStream_t stream);
