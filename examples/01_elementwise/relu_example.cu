/**
 * @file relu_example.cu
 * @brief Example demonstrating optimized ReLU activation
 *
 * This example shows:
 * - Basic ReLU implementation
 * - Vectorized ReLU using float4
 * - Performance comparison
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Include the optimized ReLU kernel
#include "../../src/01_elementwise/relu.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/**
 * @brief Naive ReLU kernel for comparison
 */
__global__ void relu_naive(const float* __restrict__ input,
                           float* __restrict__ output,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

/**
 * @brief Benchmark a kernel
 */
template <typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, int iterations = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    kernel();
    cudaDeviceSynchronize();

    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iterations;
}

/**
 * @brief Verify results match
 */
bool verify_results(const float* a, const float* b, int n, float tolerance = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Configuration
    const int N = 1 << 24;  // 16M elements
    const size_t bytes = N * sizeof(float);

    printf("=== ReLU Example ===\n");
    printf("Array size: %d elements (%.2f MB)\n", N, bytes / (1024.0f * 1024.0f));

    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float* h_output_naive = (float*)malloc(bytes);
    float* h_output_optimized = (float*)malloc(bytes);

    // Initialize input with random values
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }

    // Allocate device memory
    float *d_input, *d_output_naive, *d_output_optimized;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output_naive, bytes));
    CUDA_CHECK(cudaMalloc(&d_output_optimized, bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Kernel configurations
    const int block_size = 256;
    const int grid_size_naive = (N + block_size - 1) / block_size;
    const int grid_size_vec4 = (N / 4 + block_size - 1) / block_size;

    printf("\nKernel configurations:\n");
    printf("  Naive:     grid=%d, block=%d\n", grid_size_naive, block_size);
    printf("  Vectorized: grid=%d, block=%d\n", grid_size_vec4, block_size);

    // Benchmark naive kernel
    auto naive_kernel = [&]() {
        relu_naive<<<grid_size_naive, block_size>>>(d_input, d_output_naive, N);
    };
    float naive_time = benchmark_kernel(naive_kernel);

    // Benchmark optimized kernel
    auto optimized_kernel = [&]() {
        hpc_ai_opt::relu_forward(d_input, d_output_optimized, N);
    };
    float optimized_time = benchmark_kernel(optimized_kernel);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_optimized, d_output_optimized, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    printf("\nVerifying results...\n");
    bool correct = verify_results(h_output_naive, h_output_optimized, N);
    printf("Results %s\n", correct ? "MATCH ✓" : "MISMATCH ✗");

    // Performance results
    printf("\n=== Performance Results ===\n");
    printf("Naive ReLU:     %.4f ms\n", naive_time);
    printf("Optimized ReLU: %.4f ms\n", optimized_time);
    printf("Speedup:        %.2fx\n", naive_time / optimized_time);

    // Calculate bandwidth
    float naive_bandwidth = (2.0f * bytes) / (naive_time * 1e6);  // GB/s
    float optimized_bandwidth = (2.0f * bytes) / (optimized_time * 1e6);
    printf("\nMemory Bandwidth:\n");
    printf("  Naive:     %.2f GB/s\n", naive_bandwidth);
    printf("  Optimized: %.2f GB/s\n", optimized_bandwidth);

    // Cleanup
    free(h_input);
    free(h_output_naive);
    free(h_output_optimized);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_naive));
    CUDA_CHECK(cudaFree(d_output_optimized));

    return correct ? 0 : 1;
}
