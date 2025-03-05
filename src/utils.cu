#include "utils.cuh"

namespace radixtopk {

__global__ void init_indices_kernel(unsigned int* indices, unsigned int num_items) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_items) {
        indices[idx] = idx;
    }
}

}  // namespace radixtopk
