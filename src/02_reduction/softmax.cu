#include "softmax.cuh"
#include "../common/cuda_check.cuh"
#include "../common/reduce.cuh"
#include <cfloat>
#include <cmath>

namespace hpc::reduction {

// Online softmax kernel (single pass)
template <typename T>
__global__ void softmax_online_kernel(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       int batch, int seq_len) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const T* row_input = input + row * seq_len;
    T* row_output = output + row * seq_len;

    // Online algorithm: compute max and sum in single pass
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        float old_max = max_val;
        max_val = fmaxf(max_val, val);
        sum_exp = sum_exp * expf(old_max - max_val) + expf(val - max_val);
    }

    // Block reduction for max
    max_val = hpc::block_reduce_max(max_val);
    __shared__ float s_max, s_sum;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Recompute sum with correct max
    sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        sum_exp += expf(static_cast<float>(row_input[i]) - max_val);
    }
    sum_exp = hpc::block_reduce_sum(sum_exp);
    if (threadIdx.x == 0) s_sum = sum_exp;
    __syncthreads();
    sum_exp = s_sum;

    // Compute softmax output
    float inv_sum = 1.0f / sum_exp;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        row_output[i] = static_cast<T>(expf(static_cast<float>(row_input[i]) - max_val) * inv_sum);
    }
}

template <>
void softmax<float, SoftmaxOpt::OnlineSoftmax>(const float* input, float* output,
                                                int batch, int seq_len, cudaStream_t stream) {
    int block_size = 256;
    softmax_online_kernel<float><<<batch, block_size, 0, stream>>>(input, output, batch, seq_len);
    CUDA_CHECK_LAST();
}

template <>
void softmax<__half, SoftmaxOpt::OnlineSoftmax>(const __half* input, __half* output,
                                                 int batch, int seq_len, cudaStream_t stream) {
    int block_size = 256;
    softmax_online_kernel<__half><<<batch, block_size, 0, stream>>>(input, output, batch, seq_len);
    CUDA_CHECK_LAST();
}

} // namespace hpc::reduction
