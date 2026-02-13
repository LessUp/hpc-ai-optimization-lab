# 📖 HPC-AI-Optimization-Lab Examples

This directory contains practical examples demonstrating how to use the optimized CUDA kernels.

## Directory Structure

```
examples/
├── 01_elementwise/     # Basic elementwise operations
├── 02_reduction/       # Reduction operations (sum, max, softmax)
├── 03_gemm/           # Matrix multiplication examples
├── 04_attention/      # FlashAttention and related ops
├── python/            # Python binding examples
└── README.md          # This file
```

## Quick Start

### Building Examples

```bash
# Build all examples
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
make -j$(nproc)

# Run a specific example
./examples/01_elementwise/relu_example
```

### Running Python Examples

```bash
# Install the Python package first
pip install -e python/

# Run examples
python examples/python/basic_usage.py
python examples/python/benchmark_gemm.py
```

## Examples Overview

### 01_elementwise

| Example | Description |
|---------|-------------|
| `relu_example.cu` | ReLU activation with vectorized memory access |
| `sigmoid_example.cu` | Sigmoid with fused operations |
| `transpose_example.cu` | Matrix transpose with shared memory |

### 02_reduction

| Example | Description |
|---------|-------------|
| `sum_reduction.cu` | Parallel sum reduction |
| `softmax_example.cu` | Online softmax implementation |

### 03_gemm

| Example | Description |
|---------|-------------|
| `gemm_naive.cu` | Naive GEMM implementation |
| `gemm_tiled.cu` | Tiled GEMM with shared memory |
| `gemm_tensorcore.cu` | Tensor Core GEMM using WMMA |
| `gemm_benchmark.cu` | Performance comparison of all variants |

### 04_attention

| Example | Description |
|---------|-------------|
| `flash_attention.cu` | FlashAttention forward pass |
| `rope_example.cu` | Rotary Position Embedding |

### Python Examples

| Example | Description |
|---------|-------------|
| `basic_usage.py` | Getting started with Python bindings |
| `benchmark_gemm.py` | GEMM performance benchmarking |
| `attention_demo.py` | FlashAttention usage example |

## Performance Tips

1. **Warm-up**: Always run kernels once before timing to ensure JIT compilation
2. **Synchronization**: Use `cudaDeviceSynchronize()` for accurate timing
3. **Memory**: Pre-allocate memory outside timing loops
4. **Profiling**: Use `nsys` and `ncu` for detailed analysis

## Common Patterns

### Timing CUDA Kernels

```cpp
#include <cuda_runtime.h>

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Warm-up
kernel<<<grid, block>>>(args);
cudaDeviceSynchronize();

// Timed run
cudaEventRecord(start);
for (int i = 0; i < iterations; i++) {
    kernel<<<grid, block>>>(args);
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Average time: %.3f ms\n", ms / iterations);
```

### Error Checking

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

## License

MIT License - See [LICENSE](../LICENSE) for details.
