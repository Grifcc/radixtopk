#pragma once

namespace radixtopk {

/**
 * 初始化索引数组的简单内核
 */
__global__ void init_indices_kernel(unsigned int* indices, unsigned int num_items);

}  // namespace radixtopk
