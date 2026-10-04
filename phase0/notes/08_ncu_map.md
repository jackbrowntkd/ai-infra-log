# 笔记 08 · ncu 使用地图：能测什么、怎么查、怎么看

前置：`06_sector_and_instructions.md`（指标定义从哪查）、`07_performance_factors.md`（性能因素表）。
本机实测环境：NVIDIA T600 Laptop（sm_75）· Nsight Compute 2024.3.1。

---

## 0. 一句话

> ncu 是体检仪：**2344 个指标、23 个 section**（本机实测），
> 但日常真正要用的只有 20 来个。**学会按"我想知道什么"去找，而不是背指标名。**

---

## 1. ncu 能测什么：按硬件单元分

| 前缀 | 单元 | 本机指标数 | 回答什么问题 | 常用指标 |
|---|---|---|---|---|
| `gpu__` | 整卡 | 16 | 总耗时 | `gpu__time_duration.sum` |
| `sm__` | SM | 160 | 算力用了多少、占用率 | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| `smsp__` | SM 子分区 | 322 | 发射率、每调度器的状态 | `smsp__issue_active.avg.pct_of_peak_sustained_active` |
| `l1tex__` | L1 / TEX | 306 | **合并访存、sector 效率、L1 命中** | `l1tex__average_t_sectors_per_request_...ratio` |
| `lts__` | **L2** | **766** | 命中率、吞吐、读写 sector | `lts__t_sector_hit_rate.pct` |
| `dram__` | 显存 | 13 | **真实搬运字节数、带宽** | `dram__bytes_read.sum` / `dram__bytes_write.sum` |
| `launch__` | 启动配置 | 见 §6 第 3 条 | grid/block/寄存器/共享内存/waves | `launch__registers_per_thread` |
| `tpc__` `gpc__` `fe__` `idc__` `gr__` `sys__` `pcie__` | 其他 | 各 4~21 | 少见，按需 | — |

> L2 的指标最多（766 个），因为它的记账维度最细：
> 读 / 写 / 原子 × 命中 / 未中 × 来源单元 × 内存类型。

### 命名规则：学会这个就能自己找

```text
单元__子单元_事件.度量后缀

dram__bytes_read.sum                             DRAM 读字节数（求和）
lts__t_sector_hit_rate.pct                       L2 命中率（百分比）
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   L1 的全局读 sector 数
```

| 后缀 | 含义 |
|---|---|
| `.sum` | 求和（计数类） |
| `.avg` | 平均值 |
| `.pct` | 百分比 |
| `.ratio` | 比值 |
| `.max_rate` | 所有子单元实例里的最大值 |

**基础名不带后缀**：`ncu --query-metrics` 列的是 `lts__t_sector_hit_rate`，
后缀要用 `ncu --query-metrics-mode suffix --metrics lts__t_sector_hit_rate` 查。

---

## 2. 按"我想知道什么"找 section

| 你想知道 | 用哪个 section | 关键指标 |
|---|---|---|
| 卡算力还是带宽 | `SpeedOfLight` | `dram__throughput` vs `sm__throughput` |
| 离屋顶多远 | `SpeedOfLight_RooflineChart` | FADD/FFMA per cycle、DRAM 带宽 |
| 访存效率 | `MemoryWorkloadAnalysis`（+`_Tables`） | 各级命中率、sectors/请求、字节数 |
| 为什么发不出指令 | `SchedulerStats` + `WarpStateStats` | No Eligible、stall 原因 |
| 为什么住不满 | `Occupancy` + `LaunchStats` | 占用率、寄存器/共享内存限制 |
| 指令构成 | `InstructionStats`、`ComputeWorkloadAnalysis` | IPC、各流水线利用率 |
| **哪一行代码的锅** | `SourceCounters` | 源码行关联的 stall 与访存 |
| 随时间演化 | `PmSampling` | 采样型指标曲线 |
| 多卡互联 | `Nvlink`、`C2CLink` | — |

本机 23 个 section 的完整清单：`ncu --list-sections`。

---

## 3. 这次实验的 L2 命中率到底是什么

**指标名**：`lts__t_sector_hit_rate.pct`
**官方定义**（`--query-metrics-mode suffix` 的输出原文）：

> proportion of **L2 sector lookups** that hit

三个要点：

1. **单位是 sector（32 B）**，不是字节；
2. **分母是"L2 的 sector 查询总数"**，包含读、写、原子；
3. **写查询算命中**——写满一个 sector 的 store 不需要先从 DRAM 取数
   （笔记 05 §7 实测：`write_lookup_miss = 0`）。

### 两个坑

| 坑 | 后果 | 正确做法 |
|---|---|---|
| 写被算成命中 | vecadd 的 33 % 全是写贡献的，看着像"复用了 1/3" | 判断复用要看**只统计读**的指标 |
| 只看总和 | 分母被读/写混在一起 | 用 `lts__t_sectors_op_read_lookup_hit.sum` / `op_read.sum` |

### 本次实验的逐笔对账（64 MB 档）

```text
每次迭代的访问序列（iters = 2）：
  读a(未中)  读b(未中)  读a(中)  读b(中)  写c(中)
                                       ->  命中 3 / 总 5 = 60 %
实测：59.99 %     精确吻合
```

---

## 4. 实验怎么设计的（可复用模板）

对应代码：`phase0/cuda/cache_probe.cu`，结果见 `reports/cache_reuse_qa.md`。

| # | 原则 | 具体做法 |
|---|---|---|
| 1 | **只改一个变量** | 数据量变，逻辑工作量固定（`n x iters` 恒定） |
| 2 | **防止编译器作弊** | `volatile` + `#pragma unroll 1`，逼它每次真读内存 |
| 3 | **留单档入口** | `argv[1]` 选档，方便 ncu 单点采数（也不覆盖完整结果） |
| 4 | **时间与计数分开取** | 时间用独立运行；ncu 只取计数 |
| 5 | **用义务流量做对照** | DRAM 字节数 vs 理论值，确认没有多余搬运 |

---

## 5. 常用命令

```powershell
# 看有哪些 section / 有哪些指标
ncu --list-sections
ncu --query-metrics | Select-String "sector|hit_rate"
ncu --query-metrics-mode suffix --metrics lts__t_sector_hit_rate

# 采整段（省事，但慢：--set full 会跑几十遍）
ncu --set full -k regex:my_kernel .\a.exe

# 精准采几个数（快）
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct `
    -k regex:my_kernel --launch-skip 5 --launch-count 1 .\a.exe

# 采整个 section，并打印内部数值（roofline 的组成就是这么挖出来的）
ncu --section SpeedOfLight_RooflineChart --print-details all -k regex:my_kernel .\a.exe
```

### 选 kernel 的三个开关

| 开关 | 作用 |
|---|---|
| `-k regex:名字` | 只 profile 匹配的 kernel |
| `--launch-skip N` | 跳过前 N 次（通常是预热） |
| `--launch-count N` | 只采 N 次 |

---

## 6. 三个已知的坑

1. **被污染的是"程序自己测的时间"，不是 ncu 的 Duration**。同一档 64 MB 实测：

   | 谁测的 | 值 |
   |---|---|
   | 独立运行（CUDA event） | 1.304 ms |
   | profiler 下程序自己打印 | **18.104 ms**（放大 14 倍） |
   | **ncu 的 `gpu__time_duration.sum`** | **1.29 ms**（准） |

   原因：程序测的是"20 次 launch 的平均"，被 profile 的那一次把回放开销
   摊进了平均值；ncu 的 Duration 是 GPU 硬件时间戳，只量那一次执行。
   **可以用 ncu 的 Duration；不要用程序在 profiler 下打印的时间。**
   唯一要打折的地方：被 profile 的那次是独占运行，真实并发场景会偏乐观。
2. **`--set full` 很慢**（vecadd 跑了 30 passes）。先用 `--section`
   或几个 `--metrics` 定位，再决定要不要 full。
3. **有些指标不在 `--query-metrics` 里**（如 `launch__*`），
   它们写在 `Sections\LaunchStatistics.section` 里；直接点名
   `--metrics launch__registers_per_thread` 或采整个 section 都能拿到。

---

## 7. 下一步

- 想看"哪一行慢"，去试 `SourceCounters`（kernel 需要 `-lineinfo` 编译）
- 官方手册：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 对应清单：E4（ncu 指标体系）与 C6（计时方法论）的证据都已齐，
  等你各复述一遍就能打勾
