#include "sigmoid.cuh"
#include "../common/cuda_check.cuh"
#include <cmath>

namespace hpc::elementwise {

template <typename T>
__global__ void sigmoid_naive_kernel(const T* __restrict__ input,
                                      T* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(1.0f / (1.0f + expf(-val)));
    }
}

// Vectorized sigmoid kernel (float4)
__global__ void sigmoid_vectorized_kernel(const float* __restrict__ input,
                                           float* __restrict__ output, size_t n) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 val = reinterpret_cast<const float4*>(input)[idx / 4];
        val.x = 1.0f / (1.0f + expf(-val.x));
        val.y = 1.0f / (1.0f + expf(-val.y));
        val.z = 1.0f / (1.0f + expf(-val.z));
        val.w = 1.0f / (1.0f + expf(-val.w));
        reinterpret_cast<float4*>(output)[idx / 4] = val;
    } else {
        for (size_t i = idx; i < n; ++i) {
            float v = input[i];
            output[i] = 1.0f / (1.0f + expf(-v));
        }
    }
}

template <typename T>
__global__ void sigmoid_grid_stride_kernel(const T* __restrict__ input,
                                            T* __restrict__ output, size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(1.0f / (1.0f + expf(-val)));
    }
}

template <>
void sigmoid<float, OptLevel::Naive>(const float* input, float* output,
                                      size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    sigmoid_naive_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void sigmoid<float, OptLevel::Vectorized>(const float* input, float* output,
                                           size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n / 4 + block_size - 1) / block_size);
    sigmoid_vectorized_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void sigmoid<float, OptLevel::GridStride>(const float* input, float* output,
                                           size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = std::min(static_cast<int>((n + block_size - 1) / block_size), 1024);
    sigmoid_grid_stride_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

template <>
void sigmoid<__half, OptLevel::GridStride>(const __half* input, __half* output,
                                            size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = std::min(static_cast<int>((n + block_size - 1) / block_size), 1024);
    sigmoid_grid_stride_kernel<__half><<<grid_size, block_size, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST();
}

} // namespace hpc::elementwise
