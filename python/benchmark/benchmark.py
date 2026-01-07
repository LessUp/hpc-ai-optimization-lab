#!/usr/bin/env python3
"""
HPC-AI-Optimization-Lab Benchmark Framework

Features:
- Kernel performance comparison with PyTorch/cuBLAS
- Roofline model analysis
- HTML report generation
- Performance visualization with matplotlib
"""

import torch
from torch.utils.benchmark import Timer
import argparse
from typing import Dict, Any, Callable, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    kernel: str
    hpc_ms: float
    baseline_ms: float
    speedup: float
    tflops: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None
    arithmetic_intensity: Optional[float] = None
    efficiency: Optional[float] = None


@dataclass
class DeviceInfo:
    """GPU device information."""
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_bandwidth_gb_s: float


def get_device_info() -> DeviceInfo:
    """Get GPU device information."""
    props = torch.cuda.get_device_properties(0)

    # Calculate peak performance
    # SM count * cores per SM * 2 (FMA) * clock rate
    sm_count = props.multi_processor_count
    # Approximate cores per SM based on compute capability
    if props.major >= 8:  # Ampere+
        cores_per_sm = 128
        tensor_cores_per_sm = 4
    elif props.major >= 7:  # Volta/Turing
        cores_per_sm = 64
        tensor_cores_per_sm = 8
    else:
        cores_per_sm = 64
        tensor_cores_per_sm = 0

    clock_ghz = props.clock_rate / 1e6
    peak_fp32 = sm_count * cores_per_sm * 2 * clock_ghz / 1000  # TFLOPS
    peak_fp16 = peak_fp32 * 2  # FP16 is 2x FP32 on most GPUs

    # Memory bandwidth
    memory_clock_ghz = props.memory_clock_rate / 1e6
    bus_width_bytes = props.memory_bus_width / 8
    peak_bandwidth = 2 * memory_clock_ghz * bus_width_bytes  # GB/s (DDR)

    return DeviceInfo(
        name=props.name,
        compute_capability=(props.major, props.minor),
        total_memory_gb=props.total_memory / 1e9,
        peak_fp32_tflops=peak_fp32,
        peak_fp16_tflops=peak_fp16,
        peak_bandwidth_gb_s=peak_bandwidth,
    )


def benchmark_kernel(
    name: str,
    hpc_fn: Callable,
    baseline_fn: Callable,
    *args,
    warmup: int = 10,
    min_run_time: float = 1.0,
    flops: Optional[int] = None,
    bytes_accessed: Optional[int] = None,
    **kwargs
) -> BenchmarkResult:
    """
    Compare HPC kernel with baseline implementation.

    Args:
        name: Kernel name for reporting
        hpc_fn: HPC-optimized kernel function
        baseline_fn: Baseline function (PyTorch/cuBLAS)
        *args: Arguments to pass to kernels
        warmup: Number of warmup iterations
        min_run_time: Minimum benchmark runtime in seconds
        flops: Total floating-point operations (for TFLOPS calculation)
        bytes_accessed: Total bytes accessed (for bandwidth calculation)
        **kwargs: Keyword arguments to pass to kernels

    Returns:
        BenchmarkResult with timing and performance metrics
    """
    # Warmup
    for _ in range(warmup):
        hpc_fn(*args, **kwargs)
        baseline_fn(*args, **kwargs)

    torch.cuda.synchronize()

    # Benchmark HPC kernel
    hpc_timer = Timer(
        stmt="hpc_fn(*args, **kwargs)",
        globals={"hpc_fn": hpc_fn, "args": args, "kwargs": kwargs}
    )
    hpc_result = hpc_timer.blocked_autorange(min_run_time=min_run_time)

    # Benchmark baseline
    baseline_timer = Timer(
        stmt="baseline_fn(*args, **kwargs)",
        globals={"baseline_fn": baseline_fn, "args": args, "kwargs": kwargs}
    )
    baseline_result = baseline_timer.blocked_autorange(min_run_time=min_run_time)

    hpc_ms = hpc_result.median * 1000
    baseline_ms = baseline_result.median * 1000
    speedup = baseline_ms / hpc_ms

    # Calculate performance metrics
    tflops = None
    bandwidth = None
    arithmetic_intensity = None

    if flops is not None:
        tflops = flops / (hpc_ms * 1e-3) / 1e12

    if bytes_accessed is not None:
        bandwidth = bytes_accessed / (hpc_ms * 1e-3) / 1e9

    if flops is not None and bytes_accessed is not None and bytes_accessed > 0:
        arithmetic_intensity = flops / bytes_accessed

    return BenchmarkResult(
        kernel=name,
        hpc_ms=hpc_ms,
        baseline_ms=baseline_ms,
        speedup=speedup,
        tflops=tflops,
        bandwidth_gb_s=bandwidth,
        arithmetic_intensity=arithmetic_intensity,
    )


def compute_bandwidth(bytes_transferred: int, time_ms: float) -> float:
    """Compute bandwidth in GB/s."""
    return bytes_transferred / (time_ms * 1e-3) / 1e9


def compute_tflops(flops: int, time_ms: float) -> float:
    """Compute TFLOPS."""
    return flops / (time_ms * 1e-3) / 1e12


class RooflineAnalyzer:
    """Roofline model analysis for GPU kernels."""

    def __init__(self, device_info: Optional[DeviceInfo] = None):
        self.device_info = device_info or get_device_info()

    def analyze(self, result: BenchmarkResult) -> Dict[str, Any]:
        """
        Analyze kernel performance using roofline model.

        Returns dict with:
        - bottleneck: 'compute' or 'memory'
        - achieved_performance: actual TFLOPS
        - peak_performance: theoretical peak TFLOPS
        - efficiency: percentage of peak achieved
        """
        if result.arithmetic_intensity is None or result.tflops is None:
            return {"error": "Missing arithmetic intensity or TFLOPS data"}

        ai = result.arithmetic_intensity
        achieved_tflops = result.tflops

        # Ridge point: where compute and memory rooflines meet
        ridge_point = self.device_info.peak_fp32_tflops / self.device_info.peak_bandwidth_gb_s

        # Determine bottleneck
        if ai < ridge_point:
            # Memory bound
            bottleneck = "memory"
            peak_tflops = ai * self.device_info.peak_bandwidth_gb_s
        else:
            # Compute bound
            bottleneck = "compute"
            peak_tflops = self.device_info.peak_fp32_tflops

        efficiency = (achieved_tflops / peak_tflops) * 100 if peak_tflops > 0 else 0

        return {
            "bottleneck": bottleneck,
            "achieved_tflops": achieved_tflops,
            "peak_tflops": peak_tflops,
            "efficiency": efficiency,
            "arithmetic_intensity": ai,
            "ridge_point": ridge_point,
        }

    def plot_roofline(
        self,
        results: List[BenchmarkResult],
        output_path: str = "roofline.png",
        title: str = "Roofline Analysis"
    ):
        """Generate roofline plot for multiple kernels."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            print("Warning: matplotlib/numpy not available, skipping roofline plot")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Roofline boundaries
        ai_range = np.logspace(-2, 4, 1000)
        peak_compute = self.device_info.peak_fp32_tflops
        peak_bandwidth = self.device_info.peak_bandwidth_gb_s

        # Memory roof
        memory_roof = ai_range * peak_bandwidth

        # Compute roof
        compute_roof = np.full_like(ai_range, peak_compute)

        # Combined roofline
        roofline = np.minimum(memory_roof, compute_roof)

        # Plot roofline
        ax.loglog(ai_range, roofline, 'b-', linewidth=2, label='Roofline')
        ax.loglog(ai_range, memory_roof, 'b--', alpha=0.5, label='Memory Bound')
        ax.axhline(y=peak_compute, color='b', linestyle=':', alpha=0.5, label='Compute Bound')

        # Plot kernel results
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for result, color in zip(results, colors):
            if result.arithmetic_intensity and result.tflops:
                ax.scatter(
                    result.arithmetic_intensity,
                    result.tflops,
                    s=200,
                    c=[color],
                    marker='o',
                    label=result.kernel,
                    zorder=5
                )

        # Ridge point
        ridge_point = peak_compute / peak_bandwidth
        ax.axvline(x=ridge_point, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(
            f'Ridge Point\n({ridge_point:.1f} FLOP/B)',
            xy=(ridge_point, peak_compute * 0.5),
            fontsize=9,
            ha='center'
        )

        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
        ax.set_title(f'{title}\n{self.device_info.name}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.01, 10000)
        ax.set_ylim(0.01, peak_compute * 2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Roofline plot saved to {output_path}")


def print_results(results: List[BenchmarkResult], device_info: Optional[DeviceInfo] = None):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 90)
    if device_info:
        print(f"Device: {device_info.name}")
        print(f"Peak FP32: {device_info.peak_fp32_tflops:.1f} TFLOPS | "
              f"Peak Bandwidth: {device_info.peak_bandwidth_gb_s:.0f} GB/s")
        print("=" * 90)

    header = f"{'Kernel':<25} {'HPC (ms)':<10} {'Base (ms)':<10} {'Speedup':<10}"
    if any(r.tflops for r in results):
        header += f"{'TFLOPS':<10}"
    if any(r.bandwidth_gb_s for r in results):
        header += f"{'BW (GB/s)':<12}"
    print(header)
    print("-" * 90)

    for r in results:
        line = f"{r.kernel:<25} {r.hpc_ms:<10.4f} {r.baseline_ms:<10.4f} {r.speedup:<10.2f}x"
        if r.tflops:
            line += f"{r.tflops:<10.2f}"
        if r.bandwidth_gb_s:
            line += f"{r.bandwidth_gb_s:<12.1f}"
        print(line)
    print("=" * 90)


def generate_html_report(
    results: List[BenchmarkResult],
    device_info: DeviceInfo,
    output_path: str = "benchmark_report.html",
    roofline_image: Optional[str] = None
):
    """Generate HTML benchmark report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HPC-AI-Optimization-Lab Benchmark Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .device-info {{ background: #e8f5e9; padding: 15px; border-radius: 4px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .speedup {{ font-weight: bold; }}
        .speedup.good {{ color: #4CAF50; }}
        .speedup.bad {{ color: #f44336; }}
        .roofline {{ text-align: center; margin: 30px 0; }}
        .roofline img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .footer {{ text-align: center; color: #888; margin-top: 30px; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 HPC-AI-Optimization-Lab Benchmark Report</h1>
        <p>Generated: {timestamp}</p>

        <div class="device-info">
            <h3>Device Information</h3>
            <p><strong>GPU:</strong> {device_info.name}</p>
            <p><strong>Compute Capability:</strong> {device_info.compute_capability[0]}.{device_info.compute_capability[1]}</p>
            <p><strong>Memory:</strong> {device_info.total_memory_gb:.1f} GB</p>
            <p><strong>Peak FP32:</strong> {device_info.peak_fp32_tflops:.1f} TFLOPS</p>
            <p><strong>Peak Bandwidth:</strong> {device_info.peak_bandwidth_gb_s:.0f} GB/s</p>
        </div>

        <h2>Benchmark Results</h2>
        <table>
            <tr>
                <th>Kernel</th>
                <th>HPC Time (ms)</th>
                <th>Baseline (ms)</th>
                <th>Speedup</th>
                <th>TFLOPS</th>
                <th>Bandwidth (GB/s)</th>
            </tr>
"""

    for r in results:
        speedup_class = "good" if r.speedup >= 1.0 else "bad"
        tflops_str = f"{r.tflops:.2f}" if r.tflops else "-"
        bw_str = f"{r.bandwidth_gb_s:.1f}" if r.bandwidth_gb_s else "-"

        html += f"""            <tr>
                <td>{r.kernel}</td>
                <td>{r.hpc_ms:.4f}</td>
                <td>{r.baseline_ms:.4f}</td>
                <td class="speedup {speedup_class}">{r.speedup:.2f}x</td>
                <td>{tflops_str}</td>
                <td>{bw_str}</td>
            </tr>
"""

    html += """        </table>
"""

    if roofline_image and os.path.exists(roofline_image):
        html += f"""
        <h2>Roofline Analysis</h2>
        <div class="roofline">
            <img src="{roofline_image}" alt="Roofline Plot">
        </div>
"""

    html += """
        <div class="footer">
            <p>HPC-AI-Optimization-Lab | High-Performance CUDA Kernels for AI/ML</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")


def plot_speedup_chart(
    results: List[BenchmarkResult],
    output_path: str = "speedup_chart.png",
    title: str = "Kernel Speedup vs Baseline"
):
    """Generate speedup bar chart."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping speedup chart")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    kernels = [r.kernel for r in results]
    speedups = [r.speedup for r in results]
    colors = ['#4CAF50' if s >= 1.0 else '#f44336' for s in speedups]

    bars = ax.bar(kernels, speedups, color=colors)

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(
            f'{speedup:.2f}x',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline')
    ax.set_xlabel('Kernel', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Speedup chart saved to {output_path}")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="HPC-AI-Optimization-Lab Benchmark Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --suite gemm --sizes 1024,2048,4096
  python benchmark.py --suite all --output results.json --html report.html
  python benchmark.py --roofline --output roofline.png
        """
    )
    parser.add_argument("--suite", type=str, default="all",
                        choices=["all", "gemm", "elementwise", "reduction", "attention"],
                        help="Benchmark suite to run")
    parser.add_argument("--sizes", type=str, default="1024,2048,4096",
                        help="Comma-separated list of sizes to benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--html", type=str, help="Output HTML report file")
    parser.add_argument("--roofline", action="store_true", help="Generate roofline plot")
    parser.add_argument("--chart", action="store_true", help="Generate speedup chart")
    args = parser.parse_args()

    print("=" * 60)
    print("HPC-AI-Optimization-Lab Benchmark Framework")
    print("=" * 60)

    # Get device info
    device_info = get_device_info()
    print(f"\nDevice: {device_info.name}")
    print(f"Compute Capability: {device_info.compute_capability[0]}.{device_info.compute_capability[1]}")
    print(f"Peak FP32: {device_info.peak_fp32_tflops:.1f} TFLOPS")
    print(f"Peak Bandwidth: {device_info.peak_bandwidth_gb_s:.0f} GB/s")

    # Parse sizes
    sizes = [int(s) for s in args.sizes.split(",")]

    # Placeholder for actual benchmarks
    # In a real implementation, this would import and run the actual kernels
    print("\nNote: Run specific benchmark scripts for detailed results:")
    print("  - python benchmark/gemm_benchmark.py")
    print("  - python benchmark/attention_benchmark.py")
    print("  - python benchmark/elementwise_benchmark.py")

    # Example results (placeholder)
    results = []

    if results:
        print_results(results, device_info)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "device": asdict(device_info),
                    "results": [asdict(r) for r in results],
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            print(f"Results saved to {args.output}")

        if args.html:
            roofline_img = "roofline.png" if args.roofline else None
            generate_html_report(results, device_info, args.html, roofline_img)

        if args.roofline:
            analyzer = RooflineAnalyzer(device_info)
            analyzer.plot_roofline(results)

        if args.chart:
            plot_speedup_chart(results)


if __name__ == "__main__":
    main()
