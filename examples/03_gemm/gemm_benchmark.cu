/**
 * @file gemm_benchmark.cu
 * @brief Comprehensive GEMM benchmark comparing all optimization levels
 *
 * This example demonstrates:
 * - 7-step GEMM optimization progression
 * - Performance comparison with cuBLAS
 * - Roofline analysis
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

// Include GEMM implementations
#include "../../src/03_gemm/gemm_naive.cuh"
#include "../../src/03_gemm/gemm_tiled.cuh"
#include "../../src/03_gemm/gemm_vectorized.cuh"
#include "../../src/03_gemm/gemm_double_buffer.cuh"
#include "../../src/03_gemm/gemm_warp_tiling.cuh"
#include "../../src/03_gemm/gemm_wmma.cuh"
#include "../../src/03_gemm/gemm_mma.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__);    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

struct BenchmarkResult {
    std::string name;
    float time_ms;
    float tflops;
    float efficiency;  // vs cuBLAS
};

/**
 * @brief Calculate TFLOPS for GEMM
 */
float calculate_tflops(int M, int N, int K, float time_ms) {
    // GEMM: 2*M*N*K FLOPs (multiply-add)
    double flops = 2.0 * M * N * K;
    return (flops / (time_ms * 1e-3)) / 1e12;
}

/**
 * @brief Benchmark a GEMM kernel
 */
template <typename GemmFunc>
float benchmark_gemm(GemmFunc gemm_func, int iterations = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    gemm_func();
    cudaDeviceSynchronize();

    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        gemm_func();
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
 * @brief Run all GEMM benchmarks for a given size
 */
void run_benchmarks(int M, int N, int K) {
    printf("\n========================================\n");
    printf("GEMM Benchmark: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================\n");

    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    // Initialize with random values
    srand(42);
    for (auto& val : h_A) val = (float)rand() / RAND_MAX;
    for (auto& val : h_B) val = (float)rand() / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    std::vector<BenchmarkResult> results;

    // cuBLAS baseline
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    auto cublas_gemm = [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    };
    float cublas_time = benchmark_gemm(cublas_gemm);
    float cublas_tflops = calculate_tflops(M, N, K, cublas_time);
    results.push_back({"cuBLAS", cublas_time, cublas_tflops, 100.0f});

    // Step 1: Naive GEMM
    auto naive_gemm = [&]() {
        hpc_ai_opt::gemm_naive(d_A, d_B, d_C, M, N, K);
    };
    float naive_time = benchmark_gemm(naive_gemm);
    float naive_tflops = calculate_tflops(M, N, K, naive_time);
    results.push_back({"Step 1: Naive", naive_time, naive_tflops,
                       (cublas_time / naive_time) * 100.0f});

    // Step 2: Tiled GEMM
    auto tiled_gemm = [&]() {
        hpc_ai_opt::gemm_tiled(d_A, d_B, d_C, M, N, K);
    };
    float tiled_time = benchmark_gemm(tiled_gemm);
    float tiled_tflops = calculate_tflops(M, N, K, tiled_time);
    results.push_back({"Step 2: Tiled", tiled_time, tiled_tflops,
                       (cublas_time / tiled_time) * 100.0f});

    // Step 3: Vectorized
    auto vec_gemm = [&]() {
        hpc_ai_opt::gemm_vectorized(d_A, d_B, d_C, M, N, K);
    };
    float vec_time = benchmark_gemm(vec_gemm);
    float vec_tflops = calculate_tflops(M, N, K, vec_time);
    results.push_back({"Step 3: Vectorized", vec_time, vec_tflops,
                       (cublas_time / vec_time) * 100.0f});

    // Step 4: Double Buffer
    auto db_gemm = [&]() {
        hpc_ai_opt::gemm_double_buffer(d_A, d_B, d_C, M, N, K);
    };
    float db_time = benchmark_gemm(db_gemm);
    float db_tflops = calculate_tflops(M, N, K, db_time);
    results.push_back({"Step 4: Double Buffer", db_time, db_tflops,
                       (cublas_time / db_time) * 100.0f});

    // Step 5: Warp Tiling
    auto warp_gemm = [&]() {
        hpc_ai_opt::gemm_warp_tiling(d_A, d_B, d_C, M, N, K);
    };
    float warp_time = benchmark_gemm(warp_gemm);
    float warp_tflops = calculate_tflops(M, N, K, warp_time);
    results.push_back({"Step 5: Warp Tiling", warp_time, warp_tflops,
                       (cublas_time / warp_time) * 100.0f});

    // Step 6: WMMA (Tensor Core)
    auto wmma_gemm = [&]() {
        hpc_ai_opt::gemm_wmma(d_A, d_B, d_C, M, N, K);
    };
    float wmma_time = benchmark_gemm(wmma_gemm);
    float wmma_tflops = calculate_tflops(M, N, K, wmma_time);
    results.push_back({"Step 6: WMMA", wmma_time, wmma_tflops,
                       (cublas_time / wmma_time) * 100.0f});

    // Step 7: MMA (PTX)
    auto mma_gemm = [&]() {
        hpc_ai_opt::gemm_mma(d_A, d_B, d_C, M, N, K);
    };
    float mma_time = benchmark_gemm(mma_gemm);
    float mma_tflops = calculate_tflops(M, N, K, mma_time);
    results.push_back({"Step 7: MMA (PTX)", mma_time, mma_tflops,
                       (cublas_time / mma_time) * 100.0f});

    // Print results
    printf("\n%-25s %12s %12s %12s\n", "Implementation", "Time (ms)", "TFLOPS", "vs cuBLAS");
    printf("%-25s %12s %12s %12s\n", "---------------", "---------", "------", "---------");
    for (const auto& r : results) {
        printf("%-25s %12.4f %12.2f %11.1f%%\n",
               r.name.c_str(), r.time_ms, r.tflops, r.efficiency);
    }

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Peak FP32 TFLOPS: %.2f\n",
           (prop.clockRate * 1e-6f) * prop.multiProcessorCount *
           (prop.major >= 8 ? 128 : 64) * 2 / 1000.0f);

    // Run benchmarks for different sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
    };

    for (const auto& [M, N, K] : sizes) {
        run_benchmarks(M, N, K);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
