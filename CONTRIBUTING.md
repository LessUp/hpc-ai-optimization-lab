# 贡献指南 | Contributing Guide

感谢你对 HPC-AI-Optimization-Lab 的关注！我们欢迎各种形式的贡献。

Thank you for your interest in HPC-AI-Optimization-Lab! We welcome contributions of all kinds.

## 目录 | Table of Contents

- [行为准则](#行为准则--code-of-conduct)
- [如何贡献](#如何贡献--how-to-contribute)
- [开发环境设置](#开发环境设置--development-setup)
- [代码风格](#代码风格--code-style)
- [提交规范](#提交规范--commit-guidelines)
- [Pull Request 流程](#pull-request-流程--pull-request-process)

## 行为准则 | Code of Conduct

请阅读并遵守我们的 [行为准则](CODE_OF_CONDUCT.md)。

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 如何贡献 | How to Contribute

### 报告 Bug | Reporting Bugs

1. 检查 [Issues](https://github.com/yourusername/HPC-AI-Optimization-Lab/issues) 确认 Bug 未被报告
2. 使用 Bug Report 模板创建新 Issue
3. 提供详细的复现步骤、环境信息和错误日志

### 功能请求 | Feature Requests

1. 检查 [Issues](https://github.com/yourusername/HPC-AI-Optimization-Lab/issues) 确认功能未被请求
2. 使用 Feature Request 模板创建新 Issue
3. 描述功能的用途和预期行为

### 代码贡献 | Code Contributions

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 编写代码和测试
4. 确保所有测试通过
5. 提交更改 (`git commit -m 'Add amazing feature'`)
6. 推送到分支 (`git push origin feature/amazing-feature`)
7. 创建 Pull Request

## 开发环境设置 | Development Setup

### 系统要求 | Requirements

| 依赖 | 版本要求 |
|------|----------|
| CUDA | 12.0+ (推荐 13.1+) |
| CMake | 3.24+ |
| C++ 编译器 | GCC 11+ / Clang 14+ |
| Python | 3.8+ |

### 本地构建 | Local Build

```bash
# 克隆仓库
git clone https://github.com/yourusername/HPC-AI-Optimization-Lab.git
cd HPC-AI-Optimization-Lab

# 创建构建目录
mkdir build && cd build

# 配置
cmake .. -DCMAKE_BUILD_TYPE=Debug -GNinja

# 编译
ninja

# 运行测试
ctest --output-on-failure
```

### Docker 环境 | Docker Environment

```bash
cd docker
docker-compose up -d
docker exec -it hpc-dev bash
```

### 安装 pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

## 代码风格 | Code Style

### C++/CUDA

- 遵循 `.clang-format` 配置
- 使用 Modern C++20 特性
- 使用 RAII 管理资源
- 使用 Concepts 约束模板参数
- 所有公开 API 需要 Doxygen 注释

```cpp
/**
 * @brief ReLU activation function
 * @tparam T Data type (float, __half)
 * @param input Input tensor pointer
 * @param output Output tensor pointer
 * @param n Number of elements
 * @param stream CUDA stream (optional)
 */
template<typename T>
void relu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr);
```

### Python

- 遵循 PEP 8
- 使用类型提示
- 使用 docstring 文档

### 命名规范 | Naming Conventions

| 类型 | 规范 | 示例 |
|------|------|------|
| 类/结构体 | PascalCase | `Tensor`, `CudaTimer` |
| 函数 | snake_case | `copy_from_host`, `relu` |
| 变量 | snake_case | `block_size`, `num_threads` |
| 常量 | UPPER_SNAKE_CASE | `TILE_SIZE`, `BLOCK_DIM` |
| 宏 | UPPER_SNAKE_CASE | `CUDA_CHECK` |
| 命名空间 | snake_case | `hpc::elementwise` |

## 提交规范 | Commit Guidelines

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Type

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具相关

### 示例

```
feat(gemm): add Tensor Core WMMA implementation

- Implement WMMA API for FP16 GEMM
- Add unit tests for correctness
- Update documentation

Closes #123
```

## Pull Request 流程 | Pull Request Process

1. **确保测试通过**: 所有 CI 检查必须通过
2. **更新文档**: 如果添加新功能，更新相关文档
3. **添加测试**: 新功能必须有对应的测试
4. **代码审查**: 至少需要一位维护者审查
5. **Squash 合并**: PR 将被 squash 合并到 main 分支

### PR 检查清单

- [ ] 代码遵循项目风格指南
- [ ] 所有测试通过
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] Commit message 遵循规范

## 问题？| Questions?

如有任何问题，请：
- 查看 [文档](docs/)
- 搜索 [Issues](https://github.com/yourusername/HPC-AI-Optimization-Lab/issues)
- 创建新 Issue 提问

感谢你的贡献！🚀
