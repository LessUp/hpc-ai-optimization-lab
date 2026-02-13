#include "rmsnorm.cuh"
#include "../common/cuda_check.cuh"
#include "../common/reduce.cuh"

namespace hpc::reduction {

template <typename T>
__global__ void rms_norm_kernel(const T* __restrict__ input,
                                 const T* __restrict__ gamma,
                                 T* __restrict__ output,
                                 int hidden_size, float eps) {
    int row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        sum_sq += val * val;
    }
    sum_sq = hpc::block_reduce_sum(sum_sq);
    __shared__ float s_rms;
    if (threadIdx.x == 0) s_rms = rsqrtf(sum_sq / hidden_size + eps);
    __syncthreads();
    float rms = s_rms;

    // Normalize and apply scale
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_output[i] = static_cast<T>(static_cast<float>(row_input[i]) * rms *
                                       static_cast<float>(gamma[i]));
    }
}

template <>
void rms_norm<float>(const float* input, const float* gamma,
                     float* output, int batch, int hidden_size,
                     float eps, cudaStream_t stream) {
    int block_size = 256;
    rms_norm_kernel<float><<<batch, block_size, 0, stream>>>(
        input, gamma, output, hidden_size, eps);
    CUDA_CHECK_LAST();
}

template <>
void rms_norm<__half>(const __half* input, const __half* gamma,
                      __half* output, int batch, int hidden_size,
                      float eps, cudaStream_t stream) {
    int block_size = 256;
    rms_norm_kernel<__half><<<batch, block_size, 0, stream>>>(
        input, gamma, output, hidden_size, eps);
    CUDA_CHECK_LAST();
}

} // namespace hpc::reduction
