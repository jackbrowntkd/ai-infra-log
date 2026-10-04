"""阶段 0 · 计时工具。

规则：本项目里所有快了多少的结论，都必须由这里产生。
禁止：time.time()、只测一次、不预热、不刷 L2。

用法::

    from bench import do_bench, gbps, tflops
    ms = do_bench(lambda: kernel(a, b, c), rep=200)
"""

from __future__ import annotations

import torch

_FLUSH_BUFFERS: dict[tuple[int, int], torch.Tensor] = {}


def l2_size_bytes(device: int | None = None) -> int:
    """GPU 的 L2 容量，用来决定 flush buffer 该开多大。"""
    props = torch.cuda.get_device_properties(device)
    for name in ("L2_cache_size", "l2CacheSize"):
        size = getattr(props, name, None)
        if size:
            return int(size)
    return 1 << 20


def _flush_buffer(device: int, nbytes: int) -> torch.Tensor:
    key = (device, nbytes)
    buf = _FLUSH_BUFFERS.get(key)
    if buf is None:
        buf = torch.empty(nbytes // 4, dtype=torch.int32, device=device)
        _FLUSH_BUFFERS[key] = buf
    return buf


def do_bench(
    fn,
    warmup: float = 25.0,
    rep: float = 100.0,
    *,
    flush: bool = True,
    quantiles: tuple[float, ...] = (0.5,),
) -> float | list[float]:
    """测单次 fn 的耗时，单位毫秒。

    warmup / rep 的单位是毫秒而不是次数：先跑几次估出单次耗时，再反推需要
    跑多少轮。这样快 kernel 不会测不准，慢 kernel 也不会跑到天亮。

    flush=True 时每轮之间清空 L2。不刷 L2 的话反复跑同一个 kernel，数据会
    一直留在 cache 里，测出来的是 cache 速度而不是 DRAM 速度。
    """
    fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(5):
        fn()
    end.record()
    torch.cuda.synchronize()
    est_ms = start.elapsed_time(end) / 5

    n_warmup = max(1, int(warmup / est_ms))
    n_repeat = max(1, int(rep / est_ms))

    for _ in range(n_warmup):
        fn()

    device = torch.cuda.current_device()
    buf = None
    if flush:
        flush_bytes = max(4 * l2_size_bytes(device), 1 << 22)
        buf = _flush_buffer(device, flush_bytes)

    times: list[float] = []
    for _ in range(n_repeat):
        if buf is not None:
            buf.zero_()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    picks = [times[min(int(q * len(times)), len(times) - 1)] for q in quantiles]
    return picks[0] if len(picks) == 1 else picks


def gbps(nbytes: float, ms: float) -> float:
    """吞吐量 GB/s。"""
    return nbytes / (ms * 1e-3) / 1e9


def tflops(flops: float, ms: float) -> float:
    """算力 TFLOP/s。"""
    return flops / (ms * 1e-3) / 1e12