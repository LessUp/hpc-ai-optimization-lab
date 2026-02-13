#include "layernorm.cuh"
#include "../common/cuda_check.cuh"
#include "../common/reduce.cuh"

namespace hpc::reduction {

template <typename T>
__global__ void layer_norm_kernel(const T* __restrict__ input,
                                   const T* __restrict__ gamma,
                                   const T* __restrict__ beta,
                                   T* __restrict__ output,
                                   int hidden_size, float eps) {
    int row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += static_cast<float>(row_input[i]);
    }
    sum = hpc::block_reduce_sum(sum);
    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) s_mean = sum / hidden_size;
    __syncthreads();
    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = static_cast<float>(row_input[i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = hpc::block_reduce_sum(var_sum);
    if (threadIdx.x == 0) s_inv_std = rsqrtf(var_sum / hidden_size + eps);
    __syncthreads();
    float inv_std = s_inv_std;

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (static_cast<float>(row_input[i]) - mean) * inv_std;
        row_output[i] = static_cast<T>(normalized * static_cast<float>(gamma[i]) +
                                       static_cast<float>(beta[i]));
    }
}

template <>
void layer_norm<float>(const float* input, const float* gamma, const float* beta,
                       float* output, int batch, int hidden_size,
                       float eps, cudaStream_t stream) {
    int block_size = 256;
    layer_norm_kernel<float><<<batch, block_size, 0, stream>>>(
        input, gamma, beta, output, hidden_size, eps);
    CUDA_CHECK_LAST();
}

template <>
void layer_norm<__half>(const __half* input, const __half* gamma, const __half* beta,
                        __half* output, int batch, int hidden_size,
                        float eps, cudaStream_t stream) {
    int block_size = 256;
    layer_norm_kernel<__half><<<batch, block_size, 0, stream>>>(
        input, gamma, beta, output, hidden_size, eps);
    CUDA_CHECK_LAST();
}

} // namespace hpc::reduction
