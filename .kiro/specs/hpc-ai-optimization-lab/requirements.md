# Requirements Document

## Introduction

HPC-AI-Optimization-Lab 是一本"活的"高性能 CUDA 算子开发教科书，旨在提供从 Naive 实现到极致优化的完整演进路径。项目采用现代 C++20 标准，利用 CUDA 13.1 及 Hopper/Blackwell 架构特性（TMA, WGMMA, FP8），并通过 Python Binding 与 PyTorch 进行实战验证。

## Glossary

- **Kernel**: CUDA 中在 GPU 上执行的并行函数
- **TMA (Tensor Memory Accelerator)**: Hopper 架构的张量内存加速器，用于高效数据搬运
- **WGMMA**: Warpgroup Matrix Multiply-Accumulate，Hopper 架构的矩阵乘法指令
- **FP8**: 8位浮点数格式（e4m3/e5m2），用于低精度推理
- **Coalesced_Access**: 合并访问，多个线程同时访问连续内存地址以最大化带宽
- **Bank_Conflict**: Shared Memory 的 Bank 冲突，导致串行访问
- **Tensor_Core**: NVIDIA GPU 中专门用于矩阵运算的硬件单元
- **FlashAttention**: IO-aware 的注意力机制实现，避免写入 N×N 矩阵到 HBM
- **RoPE**: Rotary Positional Embedding，旋转位置编码
- **MoE**: Mixture of Experts，混合专家模型
- **CUTLASS**: NVIDIA 官方的 CUDA 模板库，用于高性能矩阵运算
- **Nanobind**: 现代 Python/C++ 绑定库
- **Roofline_Model**: 性能分析模型，用于评估算子是访存密集还是计算密集
- **Build_System**: CMake 构建系统及相关工具链
- **Benchmark_Framework**: 性能测试框架，用于对比 Kernel 与 PyTorch 原生算子

## Requirements

### Requirement 1: 项目基础设施

**User Story:** As a 开发者, I want 一个现代化的项目基础设施, so that 我可以快速构建、测试和部署 CUDA Kernel。

#### Acceptance Criteria

1. THE Build_System SHALL 使用 CMake 3.24+ 并采用 target-based 方式配置
2. THE Build_System SHALL 使用 FetchContent 自动拉取依赖（fmt, googletest, nanobind, cutlass）
3. THE Build_System SHALL 自动检测当前显卡架构并设置对应的 -gencode 参数
4. WHEN 用户执行 cmake && make THEN THE Build_System SHALL 成功编译所有 Kernel
5. THE Docker_Environment SHALL 基于 CUDA 13.1 镜像提供可复现的开发环境
6. WHEN Docker 容器启动 THEN THE Docker_Environment SHALL 包含所有必要的编译工具和依赖

### Requirement 2: 通用工具库

**User Story:** As a 开发者, I want 一套通用的工具库, so that 我可以专注于 Kernel 优化而非重复的基础代码。

#### Acceptance Criteria

1. THE Common_Library SHALL 提供 CUDA 错误检查宏（CudaCheck）
2. THE Common_Library SHALL 提供高精度计时器（Timer）用于性能测量
3. THE Common_Library SHALL 提供 Half/BF16 类型的封装和转换工具
4. THE Common_Library SHALL 提供 RAII 风格的 Tensor 类管理 GPU 内存
5. WHEN 使用 Tensor 类分配内存 THEN THE Common_Library SHALL 在析构时自动释放资源

### Requirement 3: 访存优化基础算子

**User Story:** As a 学习者, I want 学习访存密集型算子的优化技术, so that 我可以掌握如何跑满显存带宽。

#### Acceptance Criteria

1. THE Elementwise_Module SHALL 实现 Vector Add、ReLU、Sigmoid 的 Naive 版本
2. THE Elementwise_Module SHALL 实现使用 float4/ld.global.v4 的向量化加载版本
3. THE Elementwise_Module SHALL 实现 Grid Stride Loop 以处理任意大小输入
4. THE Transpose_Kernel SHALL 实现 Naive 版本（读行写列）
5. THE Transpose_Kernel SHALL 实现使用 Shared Memory 消除 Bank Conflict 的优化版本
6. WHEN 运行优化后的 Elementwise Kernel THEN THE Kernel SHALL 达到理论显存带宽的 80% 以上

### Requirement 4: 归约与同步算子

**User Story:** As a 学习者, I want 学习归约类算子的优化技术, so that 我可以掌握线程间通信与 Warp 级原语。

#### Acceptance Criteria

1. THE Reduction_Module SHALL 实现 Softmax 的多个优化版本
2. THE Reduction_Module SHALL 实现 RMSNorm 和 LayerNorm
3. THE Reduction_Module SHALL 使用 Warp Shuffle（__shfl_down_sync）替代 Shared Memory 归约
4. THE Reduction_Module SHALL 实现 Block Reduce（CUB 风格与手写版本对比）
5. THE Softmax_Kernel SHALL 实现 Online Softmax（一次遍历，无需减去 Max 再 Exp）
6. WHEN 可用 THEN THE Reduction_Module SHALL 利用 L2 Cache 驻留优化性能

### Requirement 5: GEMM 矩阵乘法

**User Story:** As a 学习者, I want 学习 GEMM 的完整优化路径, so that 我可以掌握如何跑满 Tensor Core 算力。

#### Acceptance Criteria

1. THE GEMM_Module SHALL 实现 Step 1: Naive Global Memory 版本
2. THE GEMM_Module SHALL 实现 Step 2: Shared Memory Tiling 版本
3. THE GEMM_Module SHALL 实现 Step 3: Double Buffering 版本
4. THE GEMM_Module SHALL 实现 Step 4: Register Tiling 版本
5. THE GEMM_Module SHALL 实现 Step 5: Tensor Core WMMA API 版本
6. THE GEMM_Module SHALL 实现 Step 6: Tensor Core MMA PTX 版本
7. THE GEMM_Module SHALL 实现 Step 7: Software Pipelining 版本
8. THE GEMM_Module SHALL 提供 SGEMM、HGEMM、Int8-GEMM 的实现
9. WHEN 运行 Tensor Core 版本 THEN THE GEMM_Kernel SHALL 达到理论算力的 70% 以上

### Requirement 6: LLM 与 Transformer 专项算子

**User Story:** As a 学习者, I want 学习大模型时代的核心算子, so that 我可以优化 LLM 推理性能。

#### Acceptance Criteria

1. THE Attention_Module SHALL 实现简化版 FlashAttention Forward Pass
2. THE FlashAttention_Kernel SHALL 将 Q、K、V Tiling 到 Shared Memory
3. THE FlashAttention_Kernel SHALL 在 SRAM 中完成 Attention Score 计算，避免写入 N×N 矩阵到 HBM
4. THE Attention_Module SHALL 借鉴 FlashAttention-2 的 Warp 分配策略
5. THE RoPE_Kernel SHALL 高效处理 Complex number rotation
6. THE RoPE_Kernel SHALL 支持融合到 Attention Kernel 中
7. THE MoE_Module SHALL 实现 Routing 和 TopK 算子
8. THE MoE_Module SHALL 针对大规模数据实现快速排序/筛选

### Requirement 7: CUDA 13 与 Hopper/Blackwell 特性

**User Story:** As a 学习者, I want 学习最新的 CUDA 和 GPU 架构特性, so that 我可以掌握未来的优化方向。

#### Acceptance Criteria

1. THE TMA_Module SHALL 使用 cuda::memcpy_async 或 PTX 指令实现异步数据搬运
2. THE TMA_Module SHALL 展示如何解放 Register 和 SM，让 Copy Engine 自动搬运数据
3. THE Cluster_Module SHALL 利用 Hopper 架构的 Thread Block Clusters 特性
4. THE Cluster_Module SHALL 实现 Block 间的 Shared Memory 直接访问（Distributed Shared Memory）
5. THE FP8_Module SHALL 使用 e4m3 和 e5m2 数据类型实现 GEMM
6. THE FP8_Module SHALL 展示 FP8 Scaling 技术

### Requirement 8: 量化算子

**User Story:** As a 学习者, I want 学习量化相关的算子, so that 我可以优化模型推理的内存和计算效率。

#### Acceptance Criteria

1. THE Quantization_Module SHALL 实现 Weight-Only Dequantization
2. THE Quantization_Module SHALL 实现 FP8 Scaling 算子
3. THE Quantization_Module SHALL 支持 INT8 量化和反量化
4. WHEN 执行量化 Kernel THEN THE Quantization_Module SHALL 保持数值精度在可接受范围内

### Requirement 9: 现代 C++ 编码规范

**User Story:** As a 开发者, I want 项目遵循现代 C++ 编码规范, so that 代码具有可读性、可维护性和教学价值。

#### Acceptance Criteria

1. THE Codebase SHALL 强制使用 C++20 标准
2. THE Codebase SHALL 使用 Concepts 约束 Kernel 模板参数
3. THE Codebase SHALL 大量使用 constexpr 计算 Grid/Block 大小和 Shared Memory 布局
4. THE Codebase SHALL 在 Host 代码中使用 auto 和 lambda 简化 CUDA API 调用
5. THE Codebase SHALL 避免使用裸指针（malloc/cudaMalloc），使用 RAII 封装
6. WHEN 定义 Kernel 模板 THEN THE Codebase SHALL 使用 requires 子句限制类型

### Requirement 10: Python 绑定与 Benchmark

**User Story:** As a 用户, I want 通过 Python 调用 CUDA Kernel 并与 PyTorch 对比, so that 我可以验证优化效果。

#### Acceptance Criteria

1. THE Python_Binding SHALL 使用 Nanobind 实现零拷贝绑定
2. THE Python_Binding SHALL 支持直接传入 PyTorch Tensor
3. THE Benchmark_Framework SHALL 使用 torch.utils.benchmark 自动对比 Kernel 与 PyTorch 原生算子
4. THE Benchmark_Framework SHALL 输出耗时、TFLOPS 和 Bandwidth 指标
5. WHEN 运行 Benchmark THEN THE Benchmark_Framework SHALL 生成可视化的性能报告

### Requirement 11: 测试与文档

**User Story:** As a 开发者, I want 完善的测试和文档, so that 我可以确保代码正确性并学习优化原理。

#### Acceptance Criteria

1. THE Test_Suite SHALL 使用 GoogleTest 框架编写单元测试
2. THE Test_Suite SHALL 覆盖所有 Kernel 的正确性验证
3. THE Documentation SHALL 为每个 Case 提供 Nsight Compute 的 Profiling 截图
4. THE Documentation SHALL 解释 Occupancy、Memory Throughput、Compute Throughput 等指标
5. THE Documentation SHALL 使用 Roofline Model 图表证明优化效果
6. THE README SHALL 提供详细的教程和性能报告

### Requirement 12: 卷积算子

**User Story:** As a 学习者, I want 学习卷积算子的优化技术, so that 我可以优化 CNN 模型的推理性能。

#### Acceptance Criteria

1. THE Convolution_Module SHALL 实现 Implicit GEMM 卷积
2. THE Convolution_Module SHALL 实现 Winograd 卷积
3. THE Convolution_Module SHALL 支持常见的卷积参数（stride、padding、dilation）
4. WHEN 运行优化后的卷积 Kernel THEN THE Kernel SHALL 性能接近 cuDNN
