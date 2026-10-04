"""阶段 0 · Roofline 图（双面板：DRAM 口径 / L2 口径）。

左边是标准 roofline：横轴 AI = FLOP / **DRAM 字节**。
右边是 L2 口径：横轴 AI = FLOP / **L2 字节**。

为什么要两张：同一个 kernel 在两个口径下的 AI 能差几十倍。
"点离屋顶很远"经常不是没优化好，而是**屋顶画错了层级**
（见 notes/05 §10 与盲区 2）。

字节数由 ncu 实测后回填进 `measured/*.json` 的
`dram_bytes` / `l2_bytes` 两个字段；缺失的点只在左图出现。

运行::

    python phase0/roofline.py

读 phase0/hardware.json 和 phase0/measured/*.json，
输出 phase0/reports/roofline.png。
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def load_hardware() -> dict:
    return json.loads((HERE / "hardware.json").read_text(encoding="utf-8"))


def load_points() -> list[dict]:
    """给每个点补上 ai_dram / ai_l2 两个口径的算术强度。

    ai_l2 用同一个 FLOP 分子换算： ai_l2 = ai_dram * dram_bytes / l2_bytes
    """
    pts: list[dict] = []
    for f in sorted((HERE / "measured").glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for p in data.get("points", []):
            # 纯访存 kernel 的算力用 gbps * ai 推出来
            gflops = p["gflops"] or p["gbps"] * p["ai"]
            db, lb = p.get("dram_bytes"), p.get("l2_bytes")
            ai_l2 = p["ai"] * db / lb if (db and lb) else None
            pts.append({**p, "gflops": gflops, "ai_dram": p["ai"],
                        "ai_l2": ai_l2, "source": f.stem})
    return pts


def draw_panel(ax, pts, bw_gbps, peak_gflops, xkey, xlabel, title):
    ridge = peak_gflops / bw_gbps
    x = np.logspace(-2.5, 3.2, 400)
    ax.plot(x, bw_gbps * x, "b-", lw=2, label=f"带宽屋顶 {bw_gbps:.0f} GB/s")
    ax.plot(x, np.full_like(x, peak_gflops), "r-", lw=2,
            label=f"算力屋顶 {peak_gflops:.0f} GFLOP/s")
    ax.axvline(ridge, color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(ridge * 1.15, peak_gflops * 0.28, f"转折点 {ridge:.1f}",
            fontsize=8, color="gray")

    colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    # 多个点的 AI 可能挤在一起（换口径后尤其明显），标签按序错开
    offsets = [(9, 7), (9, -14), (-58, 8), (-58, -14), (9, 20), (-58, 20)]
    for i, p in enumerate(pts):
        xv = p.get(xkey)
        if xv is None:
            continue
        c = colors[i % len(colors)]
        ax.plot(xv, p["gflops"], "o", color=c, ms=8, zorder=5)
        ax.annotate(p["name"], (xv, p["gflops"]), textcoords="offset points",
                    xytext=offsets[i % len(offsets)], fontsize=8, color=c)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 2e3)
    ax.set_ylim(1, peak_gflops * 2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("性能  (GFLOP/s)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)


def main() -> None:
    hw = load_hardware()
    bw_gbps = hw["bandwidth"]["read_gbps"]
    peak_gflops = hw["matmul"]["fp32"]["tflops"] * 1000.0
    l2_gbps = hw.get("cache", {}).get("l2_gbps", 289.0)
    pts = load_points()

    print(f"DRAM 带宽上限 {bw_gbps:.1f} GB/s   算力上限 {peak_gflops:.1f} GFLOP/s"
          f"   转折点 {peak_gflops / bw_gbps:.1f}")
    print(f"L2   服务速率 {l2_gbps:.1f} GB/s   转折点 {peak_gflops / l2_gbps:.1f}")
    print()
    print(f"  {'point':<22} {'AI(DRAM)':>10} {'AI(L2)':>10} {'GFLOP/s':>9}")
    for p in pts:
        ai_l2 = f"{p['ai_l2']:>10.4f}" if p["ai_l2"] is not None else "       n/a"
        print(f"  {p['name']:<22} {p['ai_dram']:>10.4f} {ai_l2} {p['gflops']:>9.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.4))
    draw_panel(ax1, pts, bw_gbps, peak_gflops, "ai_dram",
               "算术强度  (FLOP / DRAM byte)",
               f"(a) DRAM 口径 —— 屋顶 {bw_gbps:.0f} GB/s")
    draw_panel(ax2, pts, l2_gbps, peak_gflops, "ai_l2",
               "算术强度  (FLOP / L2 byte)",
               f"(b) L2 口径 —— 屋顶 {l2_gbps:.0f} GB/s")
    fig.suptitle("Roofline — NVIDIA T600 Laptop（同一批点，两把尺子）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = ROOT / "phase0" / "reports" / "roofline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print()
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
