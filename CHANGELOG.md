# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project quality improvement: LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md
- GitHub Issue and PR templates
- Code quality tools: .clang-format, .clang-tidy, .editorconfig, pre-commit
- CI/CD with GitHub Actions
- Doxygen and Sphinx documentation configuration

## [0.1.0] - 2024-01-01

### Added
- Initial release of HPC-AI-Optimization-Lab
- **Common Library**
  - `cuda_check.cuh`: CUDA error checking macros
  - `timer.cuh`: High-precision GPU timer
  - `tensor.cuh`: RAII Tensor class with C++20 Concepts
  - `types.cuh`: Half/BF16 type wrappers

- **Elementwise Module** (`src/01_elementwise/`)
  - ReLU: Naive, Vectorized (float4), Grid Stride Loop
  - Sigmoid: Naive, Vectorized, Grid Stride Loop
  - Vector Add: Naive, Vectorized, Grid Stride Loop
  - Transpose: Naive, Shared Memory, Shared Memory + Padding

- **Reduction Module** (`src/02_reduction/`)
  - Softmax: Naive, Warp Shuffle, Online Softmax
  - LayerNorm: Warp Shuffle, Block Reduce
  - RMSNorm: Optimized implementation

- **GEMM Module** (`src/03_gemm/`)
  - Step 1: Naive Global Memory
  - Step 2: Shared Memory Tiling
  - Step 3: Double Buffering
  - Step 4: Register Tiling
  - Step 5: Tensor Core WMMA API
  - Step 6: Tensor Core MMA PTX
  - Step 7: Software Pipelining
  - Support for SGEMM, HGEMM, Int8-GEMM

- **Convolution Module** (`src/04_convolution/`)
  - Implicit GEMM convolution
  - Winograd convolution

- **Attention Module** (`src/05_attention/`)
  - FlashAttention Forward Pass
  - RoPE (Rotary Positional Embedding)
  - MoE TopK routing

- **Quantization Module** (`src/06_quantization/`)
  - Weight-Only Dequantization
  - INT8 Quantization/Dequantization
  - FP8 Scaling

- **CUDA 13 Features** (`src/07_cuda13_features/`)
  - TMA (Tensor Memory Accelerator)
  - Thread Block Clusters
  - FP8 GEMM (e4m3/e5m2)

- **Testing**
  - GoogleTest + RapidCheck property-based testing
  - Test coverage for all kernel modules

- **Documentation**
  - Comprehensive README with learning path
  - GEMM optimization guide
  - Memory optimization guide
  - Reduction optimization guide
  - FlashAttention guide
  - CUDA 13 features guide

- **Build System**
  - CMake 3.24+ with FetchContent
  - Auto GPU architecture detection
  - Docker development environment

- **Python Integration**
  - Nanobind bindings framework
  - Benchmark scripts with PyTorch comparison

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2024-01-01 | Initial release |

[Unreleased]: https://github.com/yourusername/HPC-AI-Optimization-Lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/HPC-AI-Optimization-Lab/releases/tag/v0.1.0
