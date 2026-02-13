HPC-AI-Optimization-Lab Documentation
=====================================

Welcome to the HPC-AI-Optimization-Lab documentation! This project provides
high-performance CUDA kernels optimized for AI/ML workloads.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/elementwise
   api/reduction
   api/gemm
   api/attention

.. toctree::
   :maxdepth: 2
   :caption: C++ API

   cpp_api

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
--------

HPC-AI-Optimization-Lab is a comprehensive CUDA optimization learning project
featuring:

- **7-Step GEMM Optimization**: From naive implementation to Tensor Core
- **LLM Operators**: FlashAttention, RoPE, TopK, and more
- **Modern C++20 + CUDA 13.1**: Latest language and toolkit features
- **Comprehensive Testing**: GoogleTest + RapidCheck property testing
- **Python Bindings**: Easy-to-use Python interface via Nanobind

Quick Example
-------------

.. code-block:: python

   import hpc_ai_opt as opt
   import numpy as np

   # Create input arrays
   a = np.random.randn(1024, 1024).astype(np.float32)
   b = np.random.randn(1024, 1024).astype(np.float32)

   # Perform optimized GEMM
   c = opt.gemm(a, b)

   # Use FlashAttention
   q = np.random.randn(2, 8, 512, 64).astype(np.float16)
   k = np.random.randn(2, 8, 512, 64).astype(np.float16)
   v = np.random.randn(2, 8, 512, 64).astype(np.float16)
   output = opt.flash_attention(q, k, v)

Performance
-----------

Our optimized kernels achieve near-peak performance:

+------------------+------------------+------------------+
| Kernel           | vs cuBLAS        | vs PyTorch       |
+==================+==================+==================+
| GEMM (FP16)      | 95-98%           | 1.2-1.5x faster  |
+------------------+------------------+------------------+
| FlashAttention   | N/A              | 2-3x faster      |
+------------------+------------------+------------------+
| Reduction        | 90-95%           | 1.5-2x faster    |
+------------------+------------------+------------------+

License
-------

This project is licensed under the MIT License.
