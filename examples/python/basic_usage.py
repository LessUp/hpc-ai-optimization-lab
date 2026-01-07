#!/usr/bin/env python3
"""
Basic usage example for HPC-AI-Optimization-Lab Python bindings.

This example demonstrates:
- Basic elementwise operations
- GEMM (matrix multiplication)
- FlashAttention
"""

import numpy as np

try:
    import hpc_ai_opt as opt
except ImportError:
    print("Error: hpc_ai_opt module not found.")
    print("Please install it first: pip install -e python/")
    exit(1)


def example_elementwise():
    """Demonstrate elementwise operations."""
    print("\n=== Elementwise Operations ===")

    # Create input array
    x = np.random.randn(1024, 1024).astype(np.float32)
    print(f"Input shape: {x.shape}")

    # ReLU
    y_relu = opt.relu(x)
    print(f"ReLU output range: [{y_relu.min():.4f}, {y_relu.max():.4f}]")

    # Sigmoid
    y_sigmoid = opt.sigmoid(x)
    print(f"Sigmoid output range: [{y_sigmoid.min():.4f}, {y_sigmoid.max():.4f}]")

    # Verify correctness
    np_relu = np.maximum(x, 0)
    np_sigmoid = 1 / (1 + np.exp(-x))

    relu_correct = np.allclose(y_relu, np_relu, rtol=1e-5)
    sigmoid_correct = np.allclose(y_sigmoid, np_sigmoid, rtol=1e-4)

    print(f"ReLU correct: {relu_correct}")
    print(f"Sigmoid correct: {sigmoid_correct}")


def example_gemm():
    """Demonstrate GEMM (matrix multiplication)."""
    print("\n=== GEMM (Matrix Multiplication) ===")

    # Create input matrices
    M, N, K = 1024, 1024, 1024
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    print(f"A shape: {A.shape}, B shape: {B.shape}")

    # Perform GEMM
    C = opt.gemm(A, B)
    print(f"C shape: {C.shape}")

    # Verify correctness
    C_np = A @ B
    max_diff = np.abs(C - C_np).max()
    print(f"Max difference vs NumPy: {max_diff:.6f}")

    # Benchmark
    import time

    iterations = 10

    # NumPy timing
    start = time.perf_counter()
    for _ in range(iterations):
        _ = A @ B
    np_time = (time.perf_counter() - start) / iterations * 1000
    print(f"NumPy time: {np_time:.2f} ms")

    # Our implementation timing
    start = time.perf_counter()
    for _ in range(iterations):
        _ = opt.gemm(A, B)
    opt_time = (time.perf_counter() - start) / iterations * 1000
    print(f"Optimized time: {opt_time:.2f} ms")
    print(f"Speedup: {np_time / opt_time:.2f}x")


def example_flash_attention():
    """Demonstrate FlashAttention."""
    print("\n=== FlashAttention ===")

    # Create input tensors (batch, heads, seq_len, head_dim)
    batch_size = 2
    num_heads = 8
    seq_len = 512
    head_dim = 64

    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float16)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float16)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float16)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")

    # Perform FlashAttention
    output = opt.flash_attention(Q, K, V)
    print(f"Output shape: {output.shape}")

    # Reference implementation (standard attention)
    def standard_attention(Q, K, V):
        scale = 1.0 / np.sqrt(head_dim)
        scores = np.matmul(Q.astype(np.float32), K.astype(np.float32).transpose(0, 1, 3, 2)) * scale
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        return np.matmul(attn, V.astype(np.float32)).astype(np.float16)

    output_ref = standard_attention(Q, K, V)
    max_diff = np.abs(output.astype(np.float32) - output_ref.astype(np.float32)).max()
    print(f"Max difference vs reference: {max_diff:.6f}")


def example_reduction():
    """Demonstrate reduction operations."""
    print("\n=== Reduction Operations ===")

    # Create input array
    x = np.random.randn(1024 * 1024).astype(np.float32)
    print(f"Input size: {len(x)}")

    # Sum reduction
    sum_result = opt.sum(x)
    sum_np = x.sum()
    print(f"Sum: {sum_result:.4f} (NumPy: {sum_np:.4f})")

    # Max reduction
    max_result = opt.max(x)
    max_np = x.max()
    print(f"Max: {max_result:.4f} (NumPy: {max_np:.4f})")

    # Softmax
    x_2d = np.random.randn(1024, 1024).astype(np.float32)
    softmax_result = opt.softmax(x_2d, axis=-1)
    softmax_np = np.exp(x_2d) / np.exp(x_2d).sum(axis=-1, keepdims=True)

    softmax_correct = np.allclose(softmax_result, softmax_np, rtol=1e-4)
    print(f"Softmax correct: {softmax_correct}")


def main():
    """Run all examples."""
    print("=" * 50)
    print("HPC-AI-Optimization-Lab Python Examples")
    print("=" * 50)

    # Print device info
    device_info = opt.get_device_info()
    print(f"\nDevice: {device_info['name']}")
    print(f"Compute Capability: {device_info['compute_capability']}")
    print(f"Memory: {device_info['total_memory'] / 1e9:.1f} GB")

    # Run examples
    example_elementwise()
    example_gemm()
    example_flash_attention()
    example_reduction()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
