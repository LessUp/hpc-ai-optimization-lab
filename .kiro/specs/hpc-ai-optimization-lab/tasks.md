# Implementation Plan: HPC-AI-Optimization-Lab

## Overview

本实现计划按照模块化、渐进式的方式构建项目。从基础设施开始，逐步实现各个算子模块，每个模块都包含从 Naive 到优化版本的完整实现路径。

## Tasks

- [x] 1. 项目基础设施搭建
  - [x] 1.1 创建项目目录结构
    - 创建 src/common, src/01_elementwise 等目录
    - 创建 cmake/, python/, tests/, docker/ 目录
    - _Requirements: 1.1, 1.2_

  - [x] 1.2 实现 CMakeLists.txt 主配置
    - 配置 C++20 和 CUDA 20 标准
    - 配置 FetchContent 拉取 googletest, nanobind, fmt, cutlass
    - 实现 GPU 架构自动检测
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.3 创建 Docker 开发环境
    - 基于 CUDA 13.1 镜像创建 Dockerfile
    - 安装 cmake, ninja, python3, pytorch
    - _Requirements: 1.5, 1.6_

  - [x] 1.4 验证构建系统
    - 创建简单的 hello world CUDA kernel
    - 确保 cmake && make 成功编译
    - _Requirements: 1.4_

- [x] 2. 通用工具库实现
  - [x] 2.1 实现 CudaCheck 错误检查宏
    - 创建 src/common/cuda_check.cuh
    - 实现 CUDA_CHECK 和 CUDA_CHECK_LAST 宏
    - _Requirements: 2.1_

  - [x] 2.2 实现 CudaTimer 计时器
    - 创建 src/common/timer.cuh
    - 使用 cudaEvent 实现高精度计时
    - _Requirements: 2.2_

  - [x] 2.3 编写 Timer 属性测试
    - **Property 2: Timer Non-Negativity**
    - **Validates: Requirements 2.2**

  - [x] 2.4 实现 Tensor RAII 类
    - 创建 src/common/tensor.cuh
    - 使用 C++20 Concepts 约束类型
    - 实现 move semantics，禁用 copy
    - 实现 copy_from_host, copy_to_host, to_host 方法
    - _Requirements: 2.3, 2.4, 2.5_

  - [x] 2.5 编写 Tensor 属性测试
    - **Property 1: Tensor Host-Device Round Trip**
    - **Validates: Requirements 2.3, 2.4, 2.5**

  - [x] 2.6 实现类型工具
    - 创建 src/common/types.cuh
    - 实现 Half/BF16 类型封装和转换
    - _Requirements: 2.3_

- [x] 3. Checkpoint - 基础设施验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 4. Elementwise 模块实现
  - [x] 4.1 实现 ReLU Kernel
    - 创建 src/01_elementwise/relu.cuh 和 relu.cu
    - 实现 Naive 版本
    - 实现 Vectorized (float4) 版本
    - 实现 Grid Stride Loop 版本
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.2 实现 Sigmoid Kernel
    - 创建 src/01_elementwise/sigmoid.cuh 和 sigmoid.cu
    - 实现三个优化级别
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.3 实现 Vector Add Kernel
    - 创建 src/01_elementwise/vector_add.cuh 和 vector_add.cu
    - 实现三个优化级别
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.4 编写 Elementwise 属性测试
    - **Property 3: Elementwise Operation Correctness**
    - **Validates: Requirements 3.1, 3.2, 3.3**

  - [x] 4.5 实现 Transpose Kernel
    - 创建 src/01_elementwise/transpose.cuh 和 transpose.cu
    - 实现 Naive 版本（读行写列）
    - 实现 Shared Memory 版本
    - 实现 Shared Memory + Padding 版本（消除 Bank Conflict）
    - _Requirements: 3.4, 3.5_

  - [x] 4.6 编写 Transpose 属性测试
    - **Property 4: Transpose Correctness**
    - **Property 5: Transpose Involution**
    - **Validates: Requirements 3.4, 3.5**

- [x] 5. Checkpoint - Elementwise 模块验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 6. Reduction 模块实现
  - [x] 6.1 实现 Softmax Kernel
    - 创建 src/02_reduction/softmax.cuh 和 softmax.cu
    - 实现 Naive 版本（全局原子操作）
    - 实现 Warp Shuffle 版本
    - 实现 Online Softmax 版本（单次遍历）
    - 实现 Fused 版本（L2 Cache 驻留）
    - _Requirements: 4.1, 4.3, 4.5_

  - [x] 6.2 编写 Softmax 属性测试
    - **Property 6: Softmax Output Properties**
    - **Validates: Requirements 4.1, 4.4, 4.5**

  - [x] 6.3 实现 LayerNorm Kernel
    - 创建 src/02_reduction/layernorm.cuh 和 layernorm.cu
    - 实现 Warp Shuffle 归约
    - 实现 Block Reduce
    - _Requirements: 4.2, 4.3, 4.4_

  - [x] 6.4 实现 RMSNorm Kernel
    - 创建 src/02_reduction/rmsnorm.cuh 和 rmsnorm.cu
    - 实现优化版本
    - _Requirements: 4.2_

  - [x] 6.5 编写 LayerNorm/RMSNorm 属性测试
    - **Property 7: LayerNorm/RMSNorm Output Properties**
    - **Validates: Requirements 4.2**

- [x] 7. Checkpoint - Reduction 模块验证
  - 确保所有测试通过，如有问题请询问用户


- [x] 8. GEMM 模块实现
  - [x] 8.1 实现 GEMM Step 1: Naive Global Memory
    - 创建 src/03_gemm/gemm.cuh 和 gemm.cu
    - 实现最基础的全局内存版本
    - _Requirements: 5.1_

  - [x] 8.2 实现 GEMM Step 2: Shared Memory Tiling
    - 添加 Shared Memory Tiling 优化
    - 减少全局内存访问
    - _Requirements: 5.2_

  - [x] 8.3 实现 GEMM Step 3: Double Buffering
    - 实现计算与加载重叠
    - _Requirements: 5.3_

  - [x] 8.4 实现 GEMM Step 4: Register Tiling
    - 减少 Shared Memory 压力
    - _Requirements: 5.4_

  - [x] 8.5 实现 GEMM Step 5: Tensor Core WMMA
    - 使用 WMMA API 实现 Tensor Core 版本
    - _Requirements: 5.5_

  - [x] 8.6 实现 GEMM Step 6: Tensor Core MMA PTX
    - 使用 MMA PTX 指令实现更底层控制
    - _Requirements: 5.6_

  - [x] 8.7 实现 GEMM Step 7: Software Pipelining
    - 解决指令延迟问题
    - _Requirements: 5.7_

  - [x] 8.8 实现 HGEMM 和 Int8-GEMM
    - 添加 half 和 int8 数据类型支持
    - _Requirements: 5.8_

  - [x] 8.9 编写 GEMM 属性测试
    - **Property 8: GEMM Correctness**
    - **Property 9: GEMM Associativity Approximation**
    - **Validates: Requirements 5.1-5.8**

  - [x] 8.10 实现 CUTLASS 对比 Wrapper
    - 封装 CUTLASS GEMM 用于性能对比
    - _Requirements: 5.8_

- [x] 9. Checkpoint - GEMM 模块验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 10. Attention 模块实现
  - [x] 10.1 实现 FlashAttention Forward
    - 创建 src/05_attention/flash_attention.cuh 和 flash_attention.cu
    - 实现 Q, K, V Tiling 到 Shared Memory
    - 在 SRAM 中完成 Attention Score 计算
    - 借鉴 FlashAttention-2 Warp 分配策略
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 10.2 编写 FlashAttention 属性测试
    - **Property 10: FlashAttention Correctness**
    - **Validates: Requirements 6.1**

  - [x] 10.3 实现 RoPE Kernel
    - 创建 src/05_attention/rope.cuh 和 rope.cu
    - 实现 Complex number rotation
    - 支持融合到 Attention Kernel
    - _Requirements: 6.5, 6.6_

  - [x] 10.4 编写 RoPE 属性测试
    - **Property 11: RoPE Rotation Properties**
    - **Validates: Requirements 6.5**

  - [x] 10.5 实现 MoE TopK Kernel
    - 创建 src/05_attention/topk.cuh 和 topk.cu
    - 实现大规模数据快速排序/筛选
    - _Requirements: 6.7, 6.8_

  - [x] 10.6 编写 TopK 属性测试
    - **Property 12: TopK Correctness**
    - **Validates: Requirements 6.7, 6.8**

- [x] 11. Checkpoint - Attention 模块验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 12. Convolution 模块实现
  - [x] 12.1 实现 Implicit GEMM 卷积
    - 创建 src/04_convolution/conv_implicit_gemm.cuh 和 .cu
    - 支持 stride, padding, dilation 参数
    - _Requirements: 12.1, 12.3_

  - [x] 12.2 实现 Winograd 卷积
    - 创建 src/04_convolution/conv_winograd.cuh 和 .cu
    - 实现 Winograd 变换
    - _Requirements: 12.2, 12.3_

  - [x] 12.3 编写 Convolution 属性测试
    - **Property 17: Convolution Correctness**
    - **Validates: Requirements 12.1, 12.2, 12.3**

- [x] 13. Checkpoint - Convolution 模块验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 14. Quantization 模块实现
  - [x] 14.1 实现 Weight-Only Dequantization
    - 创建 src/06_quantization/dequant.cuh 和 dequant.cu
    - _Requirements: 8.1_

  - [x] 14.2 实现 INT8 量化/反量化
    - 创建 src/06_quantization/int8_quant.cuh 和 int8_quant.cu
    - _Requirements: 8.3_

  - [x] 14.3 实现 FP8 Scaling
    - 创建 src/06_quantization/fp8_scaling.cuh 和 fp8_scaling.cu
    - _Requirements: 8.2_

  - [x] 14.4 编写 Quantization 属性测试
    - **Property 16: Quantization Round Trip**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

- [x] 15. Checkpoint - Quantization 模块验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 16. CUDA 13 特性模块实现
  - [x] 16.1 实现 TMA 异步数据搬运
    - 创建 src/07_cuda13_features/tma.cuh 和 tma.cu
    - 使用 cuda::memcpy_async 或 PTX 指令
    - _Requirements: 7.1, 7.2_

  - [x] 16.2 编写 TMA 属性测试
    - **Property 13: TMA Data Integrity**
    - **Validates: Requirements 7.1**

  - [x] 16.3 实现 Thread Block Clusters
    - 创建 src/07_cuda13_features/cluster.cuh 和 cluster.cu
    - 实现 Distributed Shared Memory 访问
    - _Requirements: 7.3, 7.4_

  - [x] 16.4 编写 Cluster 属性测试
    - **Property 14: Cluster Reduce Correctness**
    - **Validates: Requirements 7.3, 7.4**

  - [x] 16.5 实现 FP8 GEMM
    - 创建 src/07_cuda13_features/fp8_gemm.cuh 和 fp8_gemm.cu
    - 使用 e4m3 和 e5m2 数据类型
    - _Requirements: 7.5, 7.6_

  - [x] 16.6 编写 FP8 GEMM 属性测试
    - **Property 15: FP8 GEMM Bounded Error**
    - **Validates: Requirements 7.5, 7.6**

- [x] 17. Checkpoint - CUDA 13 特性模块验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 18. Python Binding 实现
  - [x] 18.1 配置 Nanobind 构建
    - 创建 python/CMakeLists.txt
    - 配置 nanobind 模块
    - _Requirements: 10.1_

  - [x] 18.2 实现 Elementwise 绑定
    - 创建 python/bindings/elementwise.cpp
    - 实现 relu, sigmoid, transpose 的 PyTorch wrapper
    - _Requirements: 10.1, 10.2_

  - [x] 18.3 实现 Reduction 绑定
    - 创建 python/bindings/reduction.cpp
    - 实现 softmax, layer_norm, rms_norm 的 PyTorch wrapper
    - _Requirements: 10.1, 10.2_

  - [x] 18.4 实现 GEMM 绑定
    - 创建 python/bindings/gemm.cpp
    - 实现 matmul 的 PyTorch wrapper
    - _Requirements: 10.1, 10.2_

  - [x] 18.5 实现 Attention 绑定
    - 创建 python/bindings/attention.cpp
    - 实现 flash_attention, rope 的 PyTorch wrapper
    - _Requirements: 10.1, 10.2_

  - [x] 18.6 编写 Python Binding 属性测试
    - **Property 18: Python Binding Zero-Copy**
    - **Validates: Requirements 10.1**

- [x] 19. Checkpoint - Python Binding 验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 20. Benchmark 框架实现
  - [x] 20.1 实现 Benchmark 脚本框架
    - 创建 python/benchmark/benchmark.py
    - 使用 torch.utils.benchmark 实现对比测试
    - _Requirements: 10.3, 10.4_

  - [x] 20.2 实现各模块 Benchmark
    - 创建 python/benchmark/bench_elementwise.py
    - 创建 python/benchmark/bench_reduction.py
    - 创建 python/benchmark/bench_gemm.py
    - 创建 python/benchmark/bench_attention.py
    - _Requirements: 10.3, 10.4_

  - [x] 20.3 实现性能报告生成
    - 输出 TFLOPS, Bandwidth, Speedup 指标
    - 生成可视化图表
    - _Requirements: 10.4, 10.5_

- [x] 21. 测试框架完善
  - [x] 21.1 配置 GoogleTest
    - 创建 tests/CMakeLists.txt
    - 配置 RapidCheck 集成
    - _Requirements: 11.1_

  - [x] 21.2 创建测试工具函数
    - 创建 tests/test_utils.hpp
    - 实现 tolerance 比较函数
    - 实现随机数据生成器
    - _Requirements: 11.1, 11.2_

- [x] 22. Final Checkpoint - 全项目验证
  - 确保所有测试通过
  - 运行完整 Benchmark 套件
  - 如有问题请询问用户

## Notes

- 每个任务都引用了具体的需求以便追溯
- Checkpoint 任务用于阶段性验证
- Property 测试验证通用正确性属性
- Unit 测试验证具体示例和边界情况
- 所有测试任务均为必需任务
