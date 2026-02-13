# Kernel Optimizations & Code Quality Improvements

**Date:** 2026-02-13

## Bug Fixes (Critical)

### Block-level reduction 修复
- **layernorm.cu**: 修复 `layer_norm_kernel` 中仅使用 warp 级归约的 bug。当 block 大小为 256 线程（8 个 warp）时，之前仅在第一个 warp 内做归约，导致 mean 和 variance 计算结果不正确。现改为 block 级归约 + shared memory 广播。
- **rmsnorm.cu**: 同样修复 `rms_norm_kernel` 中 sum_sq 的 warp-only 归约问题，改为 block 级归约。
- **softmax.cu**: 修复 `softmax_online_kernel` 中 max 和 sum 的 warp-only 归约问题，改为 block 级归约 + shared memory 广播。
- **int8_quant.cu**: 修复 `compute_scale_kernel` 中 max_abs 的 warp-only 归约问题，改为 block 级归约。

## New Features

### 公共归约工具库 `common/reduce.cuh`
- 新增 `warp_reduce_sum` / `warp_reduce_max`：warp 级 shuffle 归约原语
- 新增 `block_reduce_sum` / `block_reduce_max`：基于 shared memory 的 block 级归约，支持最多 1024 线程
- 消除了 softmax.cu、layernorm.cu、rmsnorm.cu 中的重复归约代码

### Sigmoid 向量化 kernel
- 新增 `sigmoid_vectorized_kernel`：使用 float4 向量化加载/存储，提升内存带宽利用率
- 新增 `sigmoid<float, OptLevel::Vectorized>` 模板特化

## Optimizations

### Grid-stride loop 优化
- **fp8_scaling.cu**: `fp8_scale_kernel` 改为 grid-stride loop，grid_size 上限 1024
- **int8_quant.cu**: `quantize_kernel` 改为 grid-stride loop，提升大数据量下的 occupancy

### 编译期优化
- **types.cuh**: `dtype_size()` 从 `inline` 改为 `constexpr`，支持编译期求值

### CMake 构建系统
- **CMakeLists.txt**: 将过时的 `FindCUDA/select_compute_arch` 替换为 CMake 3.24+ 原生 `CMAKE_CUDA_ARCHITECTURES=native` 自动检测
