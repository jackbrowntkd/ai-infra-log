# Kernel 性能分析报告 · 向量加

阶段 0 产出物。这份报告的结构以后每个 kernel 都照抄。

## 问题

向量加跑得够快吗？还有优化空间吗？

## 被测对象

- `phase0/cuda/vecadd.cu`，N = 64M floats（64 MiB 每数组）
- 每轮：读 2 × 256 MB + 写 256 MB = **768 MB**
- 算术强度 = 1 FLOP / 12 byte = **0.083 FLOP/byte**

## 方法

- 计时用 `common.cuh` 的 `bench_ms`：预热 10 次，CUDA event 卡住 50 次循环取平均
- **不刷 L2**。工作集 768 MB 远大于本机 1 MB L2，数据不可能整块驻留，刷不刷不影响结论
- 计数器用 `ncu --set full`，对 float4 版本做 30 遍 replay

复现命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File phase0\cuda\build.ps1 phase0\cuda\vecadd.cu
.\phase0\cuda\vecadd.exe
ncu --set full -k "regex:vecadd_float4" --launch-skip 20 --launch-count 1 .\phase0\cuda\vecadd.exe
```

## 实测吞吐

| 版本 | block | 耗时 | 带宽 |
|---|---|---|---|
| naive | 256 | 4.524 ms | 178.0 GB/s |
| float4 | 128 | 4.475 ms | 180.0 GB/s |
| float4 | 256 | 4.476 ms | 179.9 GB/s |
| float4 | 512 | 4.477 ms | 179.9 GB/s |
| float4 | 1024 | 4.475 ms | 179.9 GB/s |

正确性：16384 个采样点全部匹配，最大绝对误差 0.000e+00，PASS。

## ncu 计数器（float4, block=128）

| 指标 | 值 |
|---|---|
| Duration | 4.50 ms |
| **DRAM Throughput** | **93.31 %**  (179.11 GB/s) |
| **Compute (SM) Throughput** | **4.98 %** |
| L1/TEX Hit Rate | 0 % |
| L2 Hit Rate | 33.33 % |
| Achieved Occupancy | 94.21 % |
| Registers Per Thread | 18 |
| Waves Per SM | 1170.29 |
| Avg. Active Threads Per Warp | 32 |

## 解读

**1. 它贴着带宽屋顶。**
roofline 上 AI = 0.083，理论位置本来就在斜线上。实测 179.9 GB/s，
而机器本身的 copy 是 172 GB/s、纯读 179 GB/s、纯写 185 GB/s —— 已经到顶了。
见 `phase0/reports/roofline.png`，两个点正好压在线上。

**2. SM 几乎在摸鱼。**
Compute throughput 只有 4.98%，说明 ALU 完全不是瓶颈。
这是 memory-bound 的典型指纹：算力用不到 5%，带宽吃掉 93%。

**3. 剩下那 6.7% 去哪了？** 两个原因，都不是 bug：
- ncu 的 100% 是硬件理论峰值，不是实测可达值。实测 copy 只有 172 GB/s 就是证据。
- 尾部 `if (i < n4)` 让约 5.8% 的线程被 predicated off（30.22 / 32）。

**4. L1 命中率 0% 是正常的。**
每个字节只被读一次，没有任何复用，cache 结构帮不上忙。
L2 的 33% ≈ 1/3，正好对应「2 读 1 写」里那次写。

## 为什么向量化和 block size 都没用

naive → float4 只快了 **1.1%**，block size 从 128 扫到 1024 **完全没有区别**。

原因是这两个优化的收益都落在**指令效率**上，而瓶颈在 **DRAM 带宽**：

- float4 把访存指令数降到 1/4 —— 但 DRAM 每秒能给的字节数是固定的。
- block size 影响占用率和调度 —— 而 94.21% 的占用率早就够了。

一句话：**kernel 已经打满带宽时，所有降低指令数、提高占用率的优化，收益都是 0。**

## 可迁移的经验

- **先判断 kernel 在 roofline 的哪一侧，再决定优化方向。** 搞反就是白干。
- 判断依据：看 ncu 里 DRAM Throughput 和 Compute Throughput 谁高。
- 一个 kernel 如果 DRAM > 80% 且 Compute < 20%，就别再优化它了 ——
  除非你能**减少访存量**（kernel 融合、降精度、数据复用），而不是优化指令。