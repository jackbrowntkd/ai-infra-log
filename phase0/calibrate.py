"""阶段 0 · 硬件标定：把"这台机器到底多快"从感觉变成数字。

后面每一阶段的优化，分母都来自这里：

    利用率 = 实测性能 / 本文件测出的峰值

运行::

    python phase0/calibrate.py

输出：屏幕表格 + phase0/hardware.json。
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import torch

from bench import do_bench, gbps, l2_size_bytes, tflops

HERE = pathlib.Path(__file__).resolve().parent


def dump_device() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "cuda_cores": p.multi_processor_count * 64,
        "total_mem_GB": round(p.total_memory / 1e9, 2),
        "l2_MB": round(l2_size_bytes(0) / 1e6, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def sm_clock_mhz() -> float | None:
    """读当前 SM 频率。用来判断有没有降频。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def measure_bandwidth(nbytes: int = 256 << 20) -> dict:
    """分开测三种访存模式，因为它们的上限不一样。

    - copy  读+写，最常见的模式
    - read  只读（求和）
    - write 只写（填充），不读旧值，不受读写切换拖累

    知道这三个数，才能判断一个 kernel 是被读拖住还是被写拖住。
    """
    a = torch.randn(nbytes // 4, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)

    copy_ms = float(do_bench(lambda: b.copy_(a), rep=200))
    read_ms = float(do_bench(lambda: torch.sum(a), rep=200))
    write_ms = float(do_bench(lambda: b.fill_(1.0), rep=200))

    return {
        "copy_gbps": round(gbps(2 * nbytes, copy_ms), 1),
        "read_gbps": round(gbps(nbytes, read_ms), 1),
        "write_gbps": round(gbps(nbytes, write_ms), 1),
        "ms": {"copy": round(copy_ms, 3), "read": round(read_ms, 3), "write": round(write_ms, 3)},
    }


def measure_matmul(dtype: torch.dtype, n: int, *, tf32: bool | None = None) -> dict:
    """测 n x n 方阵乘法的算力。"""
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    if tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        a = torch.randn(n, n, dtype=dtype, device="cuda")
        b = torch.randn(n, n, dtype=dtype, device="cuda")
        ms = float(do_bench(lambda: a @ b, rep=150))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    flops = 2.0 * n**3
    return {"n": n, "ms": round(ms, 3), "tflops": round(tflops(flops, ms), 2)}


def measure_sustained(n: int = 2048, seconds: float = 6.0) -> dict:
    """测突发性能 vs 持续性能。

    笔记本 dGPU 只有 60W 功耗预算，满载几秒就会降频。所以"这块卡多快"
    这个问题，必须回答两次：第一秒多快、一直跑多快。

    只看突发值会高估机器能力；只看持续值会在优化短 kernel 时低估上限。
    """
    a = torch.randn(n, n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, n, dtype=torch.float32, device="cuda")
    flops = 2.0 * n**3

    clk0 = sm_clock_mhz()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    a @ b
    end.record()
    torch.cuda.synchronize()
    burst = tflops(flops, start.elapsed_time(end))

    clk1 = sm_clock_mhz()

    start.record()
    for _ in range(3):
        a @ b
    end.record()
    torch.cuda.synchronize()
    est_ms = start.elapsed_time(end) / 3

    count = max(1, int(seconds * 1000 / est_ms))
    start.record()
    for _ in range(count):
        a @ b
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    sustained = tflops(flops * count, total_ms)

    clk2 = sm_clock_mhz()
    return {
        "burst_tflops": round(burst, 2),
        "sustained_tflops": round(sustained, 2),
        "clock_idle_mhz": clk0,
        "clock_burst_mhz": clk1,
        "clock_sustained_mhz": clk2,
        "iterations": count,
        "elapsed_s": round(total_ms / 1000, 2),
    }


def main() -> None:
    dev = dump_device()
    max_clock = sm_clock_mhz()
    print("=" * 68)
    print("硬件")
    print("=" * 68)
    for k, v in dev.items():
        print(f"  {k:<20} {v}")

    bw = measure_bandwidth()
    print()
    print("=" * 68)
    print("显存带宽  (256 MB buffer)")
    print("=" * 68)
    print(f"  {'copy  (读+写)':<18} {bw['copy_gbps']:>8.1f} GB/s   {bw['ms']['copy']:>8.3f} ms")
    print(f"  {'read  (只读)':<18} {bw['read_gbps']:>8.1f} GB/s   {bw['ms']['read']:>8.3f} ms")
    print(f"  {'write (只写)':<18} {bw['write_gbps']:>8.1f} GB/s   {bw['ms']['write']:>8.3f} ms")

    print()
    print("=" * 68)
    print("算力")
    print("=" * 68)
    mm = {
        "fp32": measure_matmul(torch.float32, 2048),
        "fp32_tf32": measure_matmul(torch.float32, 2048, tf32=True),
        "fp16": measure_matmul(torch.float16, 2048),
        "bf16": measure_matmul(torch.bfloat16, 2048),
    }
    for name, r in mm.items():
        print(f"  {name:<12} n={r['n']:<6} {r['tflops']:>8.2f} TFLOP/s   {r['ms']:>8.3f} ms")

    peak_fp32 = mm["fp32"]["tflops"]
    peak_fp16 = mm["fp16"]["tflops"]
    has_tc = peak_fp16 > peak_fp32 * 1.5
    print()
    print("  判定：", end="")
    if has_tc:
        print(f"有 Tensor Core（fp16 是 fp32 的 {peak_fp16 / peak_fp32:.1f} 倍）")
    else:
        print(f"没有可用的 Tensor Core —— fp16 只有 fp32 的 {peak_fp16 / peak_fp32:.2f} 倍，")
        print("        甚至倒挂。说明 cuBLAS 走了无 TC 的慢路径。")
        print("        后果：本机练不了 mma / FlashAttention 这类 tensor-core kernel，")
        print("        这些内容必须上云（T4 就有 TC）或者换卡。")

    print()
    print("=" * 68)
    print("尺度效应：同样 fp32，换个矩阵大小差多少")
    print("=" * 68)
    for n in (256, 1024, 2048):
        r = measure_matmul(torch.float32, n)
        print(f"  n={n:<6} {r['tflops']:>8.2f} TFLOP/s   {r['ms']:>8.3f} ms")

    sus = measure_sustained()
    print()
    print("=" * 68)
    print("突发 vs 持续（笔记本 60W 功耗墙）")
    print("=" * 68)
    print(f"  {'第一次迭代（突发）':<22} {sus['burst_tflops']:>7.2f} TFLOP/s   时钟 {sus['clock_burst_mhz']} MHz")
    print(f"  {'连续跑 6 秒（持续）':<22} {sus['sustained_tflops']:>7.2f} TFLOP/s   时钟 {sus['clock_sustained_mhz']} MHz")
    print(f"  跑了 {sus['iterations']} 次，共 {sus['elapsed_s']} s")
    print(f"  结论：持续性能只有突发的 {sus['sustained_tflops'] / sus['burst_tflops'] * 100:.0f}%，")
    print("        所以测短 kernel 用突发值当分母，测长负载用持续值。")

    ridge_fp32 = peak_fp32 * 1e12 / (bw["read_gbps"] * 1e9)
    print()
    print("=" * 68)
    print("Roofline 转折点 (ridge point)")
    print("=" * 68)
    print(f"  fp32: {ridge_fp32:>6.1f} FLOP/byte   算术强度超过它才算 compute-bound")
    if not has_tc:
        print("  fp16: 不适用（无 Tensor Core）")

    out = {
        "device": dev,
        "bandwidth": bw,
        "matmul": mm,
        "has_tensor_core": has_tc,
        "sustained": sus,
        "ridge_point": {"fp32": round(ridge_fp32, 2)},
    }
    (HERE / "hardware.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"已写入 {HERE / 'hardware.json'}")


if __name__ == "__main__":
    main()