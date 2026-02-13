#pragma once

#include <cuda_runtime.h>

namespace hpc {

// Warp-level reduction primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction using shared memory
// Requires blockDim.x <= 1024 (max 32 warps)
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // one slot per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Only first warp does final reduction
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

} // namespace hpc
