# 🚀 HPC-AI-Optimization-Lab

<p align="center">
  <b>一本"活的"高性能 CUDA 算子开发教科书</b><br>
  <i>A Living Textbook for High-Performance CUDA Kernel Development</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-13.1+-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-20-00599C?style=flat-square&logo=cplusplus" alt="C++20">
  <img src="https://img.shields.io/badge/Architecture-Hopper%2FBlackwell-green?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License">
</p>

---

## 📖 项目简介

本项目是一个系统性的 CUDA 高性能计算教程，从 Naive 实现到极致优化，涵盖现代 AI 模型（LLM、Diffusion）所需的核心算子。

**为什么需要这个项目？**

大多数 CUDA 教程还停留在 2015 年的水平。本项目面向 **CUDA 13.1+** 和 **Hopper/Blackwell** 架构，采用 **Modern C++20** 特性编写清晰、可维护、极致高效的 Kernel。

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🔥 **渐进式优化** | 每个算子从 Naive → Vectorized → Shared Memory → Tensor Core 完整演进路径 |
| 🎯 **现代 C++20** | Concepts 类型约束、RAII 资源管理、constexpr 编译期计算 |
| ⚡ **Hopper 特性** | TMA 异步搬运、Thread Block Clusters、FP8 计算 |
| 🧠 **LLM 专项** | FlashAttention、RoPE、MoE TopK 等 Transformer 核心算子 |
| 🐍 **PyTorch 集成** | Nanobind 零拷贝绑定，无缝对接 PyTorch 生态 |
| ✅ **属性测试** | RapidCheck 属性测试确保算法正确性 |

## 📁 项目结构

```
HPC-AI-Optimization-Lab/
├── cmake/                      # 现代 CMake 工具链
├── src/
│   ├── common/                 # 🔧 基础工具库
│   │   ├── cuda_check.cuh      #    CUDA 错误检查宏
│   │   ├── timer.cuh           #    高精度 GPU 计时器
│   │   ├── tensor.cuh          #    RAII Tensor 类 (C++20 Concepts)
│   │   └── types.cuh           #    Half/BF16 类型封装
│   │
│   ├── 01_elementwise/         # 📊 访存密集型算子
│   │   ├── relu.cu/cuh         #    ReLU: Naive → Vectorized → GridStride
│   │   ├── sigmoid.cu/cuh      #    Sigmoid: 同上优化路径
│   │   ├── vector_add.cu/cuh   #    Vector Add: 合并访问示例
│   │   └── transpose.cu/cuh    #    Transpose: Shared Memory + Padding
│   │
│   ├── 02_reduction/           # 🔄 归约类算子
│   │   ├── softmax.cu/cuh      #    Softmax: Warp Shuffle + Online
│   │   ├── layernorm.cu/cuh    #    LayerNorm: Block Reduce
│   │   └── rmsnorm.cu/cuh      #    RMSNorm: 优化版本
│   │
│   ├── 03_gemm/                # ⚡ 计算密集型 (重点模块)
│   │   └── gemm.cu/cuh         #    GEMM 7步优化完整实现
│   │
│   ├── 04_convolution/         # 🖼️ 卷积算子
│   │   ├── conv_implicit_gemm.cu/cuh  # Implicit GEMM 卷积
│   │   └── conv_winograd.cu/cuh       # Winograd 变换卷积
│   │
│   ├── 05_attention/           # 🧠 LLM 核心算子
│   │   ├── flash_attention.cu/cuh     # FlashAttention Forward
│   │   ├── rope.cu/cuh                # RoPE 位置编码
│   │   └── topk.cu/cuh                # MoE TopK 路由
│   │
│   ├── 06_quantization/        # 📉 量化算子
│   │   ├── dequant.cu/cuh      #    Weight-Only 反量化
│   │   ├── int8_quant.cu/cuh   #    INT8 量化/反量化
│   │   └── fp8_scaling.cu/cuh  #    FP8 缩放
│   │
│   └── 07_cuda13_features/     # 🚀 CUDA 13 新特性
│       ├── tma.cu/cuh          #    TMA 异步数据搬运
│       ├── cluster.cu/cuh      #    Thread Block Clusters
│       └── fp8_gemm.cu/cuh     #    FP8 GEMM (e4m3/e5m2)
│
├── python/
│   ├── bindings/               # Nanobind Python 绑定
│   └── benchmark/              # PyTorch 性能对比脚本
│
├── tests/                      # GoogleTest + RapidCheck 测试
├── docker/                     # CUDA 13.1 Docker 环境
└── README.md
```

## 🛠️ 快速开始

### 环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| CUDA | 13.1+ | Hopper 特性需要 13.0+，推荐 13.1 |
| CMake | 3.24+ | 支持 CUDA 语言和 FetchContent |
| C++ 编译器 | GCC 11+ / Clang 14+ | 需要 C++20 支持 |
| Python | 3.8+ | 用于 Benchmark 和 Python 绑定 |
| PyTorch | 2.0+ | 用于性能对比测试 |

### 方式一：本地构建

```bash
# 克隆仓库
git clone https://github.com/yourusername/HPC-AI-Optimization-Lab.git
cd HPC-AI-Optimization-Lab

# 创建构建目录
mkdir build && cd build

# 配置 (自动检测 GPU 架构)
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja

# 编译
ninja

# 运行测试
ctest --output-on-failure
```

### 方式二：Docker 环境 (推荐)

```bash
# 启动开发容器
cd docker
docker-compose up -d

# 进入容器
docker exec -it hpc-ai-lab bash

# 在容器内构建
mkdir build && cd build
cmake .. -GNinja && ninja
```

### 安装 Python 绑定

```bash
# 在 build 目录下
cmake .. -DBUILD_PYTHON=ON
ninja

# 安装到 Python 环境
pip install python/

# 验证安装
python -c "import hpc_kernels; print('Success!')"
```

## 📚 优化案例详解

### 第一阶段：访存优化基础 (Memory Bound)

**目标**：跑满显存带宽

#### Case 1: Vector Add / ReLU / Sigmoid

```
优化路径: Naive → Vectorized (float4) → Grid Stride Loop
```

| 优化级别 | 技术要点 | 带宽利用率 |
|----------|----------|------------|
| Naive | 基础索引计算 | ~60% |
| Vectorized | `float4` 向量化加载 | ~85% |
| GridStride | 处理任意大小输入 | ~90% |

**关键概念**: Coalesced Access (合并访问), Memory Bandwidth Utilization

#### Case 2: Transpose (矩阵转置)

```
优化路径: Naive (读行写列) → Shared Memory → Shared Memory + Padding
```

| 优化级别 | 问题 | 解决方案 |
|----------|------|----------|
| Naive | 非合并写入 | - |
| SharedMem | Bank Conflict | 使用共享内存中转 |
| Padded | 消除 Bank Conflict | 添加 Padding (+1) |

**关键概念**: Shared Memory Banking, Bank Conflict Avoidance

---

### 第二阶段：归约与同步 (Reduction)

**目标**：掌握线程间通信与 Warp 级原语

#### Case 3: Softmax / LayerNorm / RMSNorm

```
优化路径: Naive (原子操作) → Warp Shuffle → Block Reduce → Online Softmax
```

| 优化级别 | 技术 | 性能提升 |
|----------|------|----------|
| Naive | 全局原子操作 | 基准 |
| WarpShuffle | `__shfl_down_sync` | 5-10x |
| BlockReduce | 分层归约 | 10-20x |
| OnlineSoftmax | 单次遍历 | 20-30x |

**关键概念**: Warp Shuffle, Cooperative Groups, Online Algorithm

---

### 第三阶段：计算密集型核心 (Compute Bound) ⭐

**目标**：跑满 Tensor Core 算力

#### Case 4: GEMM (通用矩阵乘) - 7步优化

这是本项目的**重中之重**，完整展示从 Naive 到 Tensor Core 的优化路径：

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7
Naive   Shared   Double   Register  WMMA     MMA PTX  Pipeline
        Tiling   Buffer   Tiling    API
```

| Step | 优化技术 | TFLOPS (FP32) | 关键改进 |
|------|----------|---------------|----------|
| 1 | Naive Global Memory | ~0.5 | 基准实现 |
| 2 | Shared Memory Tiling | ~2.0 | 减少全局内存访问 |
| 3 | Double Buffering | ~3.5 | 计算与加载重叠 |
| 4 | Register Tiling | ~6.0 | 减少 Shared Memory 压力 |
| 5 | Tensor Core (WMMA) | ~50+ | 使用 Tensor Core |
| 6 | Tensor Core (MMA PTX) | ~60+ | 更底层控制 |
| 7 | Software Pipelining | ~70+ | 隐藏指令延迟 |

**支持的数据类型**: `float`, `__half`, `int8_t`

---

### 第四阶段：LLM 与 Transformer 专项

**目标**：针对大模型时代的特殊算子

#### Case 5: FlashAttention

```cpp
// 核心思想: 在 SRAM 中完成 Attention Score 计算，避免写入 N×N 矩阵到 HBM
// Tiling Q, K, V 到 Shared Memory
// 借鉴 FlashAttention-2 的 Warp 分配策略
```

#### Case 6: RoPE (Rotary Positional Embedding)

```cpp
// 高效处理 Complex number rotation
// 支持融合到 Attention Kernel
```

#### Case 7: MoE TopK

```cpp
// 大规模数据快速排序/筛选
// 用于 Mixture of Experts 路由
```

---

### 第五阶段：CUDA 13 & Hopper/Blackwell 特性

**目标**：展示未来方向

#### Case 8: TMA (Tensor Memory Accelerator)

```cpp
// 使用 cuda::memcpy_async 或 PTX 指令
// 解放 Register 和 SM，让 Copy Engine 自动搬运数据
```

#### Case 9: Thread Block Clusters

```cpp
// 利用 Hopper 架构的 Cluster 特性
// 实现 Block 间的 Shared Memory 直接访问 (Distributed Shared Memory)
```

#### Case 10: FP8 GEMM

```cpp
// 使用 e4m3 和 e5m2 数据类型
// 下一代低精度推理
```

## 🧪 测试

本项目使用 **GoogleTest** + **RapidCheck** 进行属性测试，确保算法在所有输入下的正确性。

### 运行所有测试

```bash
cd build
ctest --output-on-failure
```

### 运行特定模块测试

```bash
# Elementwise 测试
./tests/elementwise/test_elementwise

# GEMM 测试
./tests/gemm/test_gemm

# Attention 测试
./tests/attention/test_attention
```

### 属性测试示例

```cpp
// Property 1: Tensor Host-Device Round Trip
// 对于任意 Tensor，host → device → host 应该得到相同的数据
RC_GTEST_PROP(TensorTest, RoundTrip, (std::vector<float> data)) {
    Tensor<float> tensor(data.size());
    tensor.copy_from_host(data.data());
    auto result = tensor.to_host();
    RC_ASSERT(data == result);
}

// Property 8: GEMM Correctness
// 对于任意矩阵 A, B，我们的 GEMM 结果应该与参考实现一致
RC_GTEST_PROP(GemmTest, Correctness, (Matrix A, Matrix B)) {
    auto our_result = gemm(A, B);
    auto ref_result = reference_gemm(A, B);
    RC_ASSERT(approx_equal(our_result, ref_result, 1e-5));
}
```

## 📊 性能测试

### 运行 Benchmark

```bash
cd python/benchmark

# 运行所有 Benchmark
python benchmark.py

# 运行特定模块
python benchmark.py --module gemm
python benchmark.py --module attention
```

### Benchmark 输出示例

```
================================================================================
                        HPC-AI-Optimization-Lab Benchmark
================================================================================

GEMM Performance (M=4096, N=4096, K=4096)
--------------------------------------------------------------------------------
| Implementation          | Time (ms) | TFLOPS | vs PyTorch |
|-------------------------|-----------|--------|------------|
| PyTorch (cuBLAS)        |    2.15   |  63.8  |    1.00x   |
| Naive                   |   89.32   |   1.5  |    0.02x   |
| SharedMemTiling         |   21.45   |   6.4  |    0.10x   |
| DoubleBuffer            |   12.87   |  10.7  |    0.17x   |
| RegisterTiling          |    7.23   |  19.0  |    0.30x   |
| TensorCore (WMMA)       |    2.45   |  56.1  |    0.88x   |
| TensorCore (MMA PTX)    |    2.31   |  59.5  |    0.93x   |
| SoftwarePipeline        |    2.18   |  63.0  |    0.99x   |
--------------------------------------------------------------------------------

FlashAttention Performance (batch=8, heads=32, seq_len=2048, head_dim=128)
--------------------------------------------------------------------------------
| Implementation          | Time (ms) | Memory (GB) | vs PyTorch |
|-------------------------|-----------|-------------|------------|
| PyTorch (native)        |   45.23   |    8.5      |    1.00x   |
| FlashAttention (ours)   |   12.87   |    0.5      |    3.51x   |
--------------------------------------------------------------------------------
```

### 生成性能报告

```bash
# 生成完整报告 (包含图表)
python benchmark.py --report --output report.html
```

## 📖 技术规范

### Modern C++20 标准

本项目严格遵循现代 C++ 编码规范：

#### Concepts 类型约束

```cpp
#include <concepts>

// 使用 Concept 约束 Kernel 模板参数
template<typename T>
concept CudaNumeric = std::is_same_v<T, float> || 
                      std::is_same_v<T, __half> || 
                      std::is_same_v<T, __nv_bfloat16>;

template<CudaNumeric T>
void relu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr);
```

#### RAII 资源管理

```cpp
// Tensor 类自动管理 GPU 内存
template<CudaNumeric T>
class Tensor {
public:
    explicit Tensor(size_t size);
    ~Tensor();  // 自动释放 GPU 内存
    
    // 禁用拷贝，启用移动
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    
    void copy_from_host(const T* host_data);
    void copy_to_host(T* host_data) const;
    std::vector<T> to_host() const;
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    
private:
    T* data_ = nullptr;
    size_t size_ = 0;
};
```

#### Constexpr 编译期计算

```cpp
// 编译期计算 Grid/Block 大小
constexpr int TILE_SIZE = 32;
constexpr int THREADS_PER_BLOCK = TILE_SIZE * TILE_SIZE;

// 编译期计算 Shared Memory 布局
constexpr int SMEM_SIZE = TILE_SIZE * (TILE_SIZE + 1);  // +1 避免 Bank Conflict
```

### Python 绑定 (Nanobind)

```python
import torch
import hpc_kernels

# 零拷贝：直接使用 PyTorch CUDA Tensor
x = torch.randn(1024, 1024, device='cuda')
y = torch.empty_like(x)

# 调用我们的 Kernel
hpc_kernels.elementwise.relu(x, y)

# 验证结果
assert torch.allclose(y, torch.relu(x))
```

### CMake 构建系统

```cmake
# 现代 CMake: Target-based 配置
add_library(hpc_gemm STATIC src/03_gemm/gemm.cu)
target_include_directories(hpc_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(hpc_gemm PUBLIC cxx_std_20)
target_compile_options(hpc_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# FetchContent 自动拉取依赖
include(FetchContent)
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git)
FetchContent_Declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git)
FetchContent_MakeAvailable(googletest nanobind)
```

## 🔬 Profiling 指南

### 使用 Nsight Compute

```bash
# Profile GEMM Kernel
ncu --set full -o gemm_profile ./build/tests/gemm/test_gemm

# 查看报告
ncu-ui gemm_profile.ncu-rep
```

### 关键指标解读

| 指标 | 含义 | 优化目标 |
|------|------|----------|
| Occupancy | SM 利用率 | > 50% |
| Memory Throughput | 显存带宽利用率 | > 80% (Memory Bound) |
| Compute Throughput | 计算单元利用率 | > 80% (Compute Bound) |
| SM Efficiency | 指令执行效率 | > 90% |

### Roofline Model

```
                    ┌─────────────────────────────────────┐
    TFLOPS          │                    ╱                │
       ▲            │                  ╱                  │
       │            │                ╱   Compute Bound    │
       │            │              ╱                      │
       │            │            ╱                        │
       │            │          ╱                          │
       │            │        ╱                            │
       │            │      ╱  Memory Bound                │
       │            │    ╱                                │
       │            │  ╱                                  │
       └────────────┴─────────────────────────────────────►
                              Arithmetic Intensity (FLOP/Byte)
```

## 📚 学习路线建议

### 初学者 (1-2 周)

1. **Day 1-2**: 理解 CUDA 编程模型，运行 Vector Add
2. **Day 3-4**: 学习 Coalesced Access，优化 ReLU/Sigmoid
3. **Day 5-7**: 掌握 Shared Memory，实现 Transpose
4. **Day 8-10**: 学习 Warp Shuffle，实现 Softmax
5. **Day 11-14**: 理解 GEMM 优化 Step 1-4

### 进阶 (2-4 周)

1. **Week 1**: GEMM Step 5-7 (Tensor Core)
2. **Week 2**: FlashAttention 原理与实现
3. **Week 3**: CUDA 13 新特性 (TMA, Clusters)
4. **Week 4**: 性能调优与 Profiling

### 专家 (持续学习)

- 阅读 CUTLASS 源码
- 研究 FlashAttention-2/3 论文
- 跟踪 NVIDIA 最新架构特性

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范

- 遵循 Modern C++20 风格
- 所有 Kernel 必须有对应的属性测试
- 提交前运行 `ctest` 确保测试通过

## 🙏 致谢

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) - 高性能 GEMM 模板库
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - IO-aware Attention
- [Nanobind](https://github.com/wjakob/nanobind) - 高效 Python 绑定
- [RapidCheck](https://github.com/emil-e/rapidcheck) - C++ 属性测试框架

## 📬 联系方式

如有问题或建议，请提交 [Issue](https://github.com/yourusername/HPC-AI-Optimization-Lab/issues)。

---

<p align="center">
  <b>Happy CUDA Hacking! 🚀</b>
</p>
