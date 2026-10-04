# 学习路线图 · CUDA 编程与 GPU 硬件

> **这份文件解决的问题**：在开始学之前就知道**范围有多大**、**现在在第几条**、**什么时候算学完**。
>
> 以前的方式是"你问一个点，我答一个点，顺手再冒出三个新名词"。从这份文件开始改成：
> **清单先定，进度可数，新东西先登记再学。**

---

## 0. 三条规矩

1. **只勾有产出物的条目。** 看完 ≠ 学会。每条的 `验收` 是能自己复现的证据
   （一个脚本、一份报告、一段能讲清楚的解释）。
2. **清单外的新名词，先登记再学。** 任何不在本清单里的概念，必须先进
   §4 名词表并标注「计划外补充」，由你决定加不加进清单——不允许它悄悄出现。
3. **进度 = 已勾条数 / 总条数**，写在 §1 的总览里，随时可查。

---

## 1. 进度总览

| 关卡 | 主题 | 条目数 | 已完成 | 状态 | 对应笔记 |
|---|---|---|---|---|---|
| **L1** | **CUDA 编程与 GPU 硬件** | **45** | **28** | 进行中 62% | 01–09 |
| L2 | Kernel 工程化（共享内存 / 指令级） | 12 | 0 | 未开始（触发条件见 §3） | — |
| L3 | 并行训练与通信 | — | 0 | 未展开 | — |
| L4 | 推理系统 | — | 0 | 未展开 | — |
| L5 | 编译与图优化 / 系统规模化 | — | 0 | 未展开 | — |

**L1 的终点线**：45 条全勾 → CUDA 编程与 GPU 硬件这一节学完，进入 L2。
还剩 19 条，按每周 2 条的节奏约 **9–10 周**。
（条目分布：A 编程模型 7 · B 硬件实现 7 · C 性能工程 10 · D 语言扩展 10 · E 工具链 6 · F 实战产出 5）

---

## 2. L1 详细清单（CUDA 编程与 GPU 硬件）

格式：`-[x] 编号 名称 —— 产出物 · 验收 · 出处`
出处里的 `PG §x.y` = CUDA C++ Programming Guide 对应章节（可在 §5 给出的地址查到）。

### A. 编程模型

- [x] **A1 线程层级与物理映射链** —— `notes/03 §十一` · 能不看资料画出 线程→warp→block→SM→SMSP，并说出 `blockIdx` 从哪来 · PG §2.1–2.2, §4.1
- [x] **A2 block / grid 的维度与上限** —— `cuda/device_query.cu` · 能说出本机每 SM 1024 线程 / 32 warp 的来源 · PG §2.2
- [x] **A3 内存层级总览** —— `notes/01, 02` · 能报出寄存器/L1/共享/L2/DRAM 五级容量与相对距离 · PG §2.3
- [x] **A4 编程模型 vs 硬件的落差** —— `notes/03 §十` · 能解释"你写的是每线程，跑的是 warp+SM" · PG §2.1
- [ ] **A5 异构编程模型**（host/device、统一内存） —— 产出：一个用 UVA 的完整例子 · PG §2.4
- [ ] **A6 异步 SIMT 编程模型**（async copy / barrier 的模型层） —— PG §2.5
- [x] **A7 计算能力与兼容性** —— `notes/02` · 能说出 sm_75 意味着什么、哪些特性缺席 · PG §2.6, §6

### B. 硬件实现

- [x] **B1 SIMT 与 warp 齐步走** —— `notes/03 §四` · 能解释分叉为什么拖慢全班 · PG §4.1
- [x] **B2 硬件多线程与延迟隐藏** —— `notes/01` · 能用 `No Eligible 97%` 解释"人手足但在等料" · PG §4.2
- [x] **B3 SM 微架构**（SMSP / lane / 调度器） —— `notes/03 §一,§五` · PG §4.1
- [x] **B4 寄存器文件与占用率** —— `cuda/reg_width.cu`, `notes/03` · 能解释寄存器如何限制入住率 · PG §4.1
- [x] **B5 sector 与内存事务记账** —— `notes/06` · 能算出 `sectors × 32 == 字节数` 才算合并完美 · PG §5.3
- [x] **B6 L2 的角色与写回行为** —— `notes/06 §3, §「为什么请求数翻4倍」`, `reports/cache_reuse_qa.md §6` · 能解释"为什么写发生在 L2""写是先记在 L2、淘汰时才回 DRAM""命中率 33 % 是读写比例的镜像" · PG §2.3, NCU 文档
- [ ] **B7 缓存提示指令的边界**（`.cg/.cs/.wt/prefetch/discard`） —— **证据已齐**（7 个变体 PTX/SASS 对照表在 `reports/`），**待复述后打勾** · 能说出本机支持哪几种、`discard` 为什么被拒绝 · PG §7.11–7.12, PTX ISA

### C. 性能工程

- [x] **C1 先测量后优化** —— `notes/05`, `reports/vecadd_profile.md` · PG §5.1
- [x] **C2 roofline 与算术强度** —— `notes/05`, `reports/roofline.png` · 能判断一个 kernel 卡算力还是带宽 · PG §5.2
- [x] **C3 合并访存** —— `notes/01`, `notes/06` · **产出已达标**（sector 数 = 字节数/32）· PG §5.3
- [x] **C4 向量化与访存宽度** —— `notes/06`, `reports/membound_sweep.txt` · 能解释 float3 为什么指令多却不慢 · PG §5.3
- [x] **C5 占用率与寄存器压力** —— `notes/03` · PG §5.2
- [x] **C6 计时方法论**（do_bench / event / 刷 L2） —— `bench.py`, `common.cuh`, `AGENTS.md` · 能说出什么时候必须刷 L2 · —
- [ ] **C7 最大化指令吞吐** —— 产出：一个 SM 利用率 > 40% 的 kernel + ncu 报告 · PG §5.4
- [ ] **C8 减少内存抖动**（Minimize Memory Thrashing） —— PG §5.5
- [x] **C9 共享内存与 tiling** —— `cuda/sgemm.cu`（naive / tiled / 寄存器分块 / reg2 四版）+ `reports/sgemm_qa.md` · 能解释"为什么 tiling 只快 1.4 倍"（因为 DRAM 本来就不是瓶颈），以及"为什么 v3 靠修访存模式又快了 82%" · PG §5.3
- [ ] **C10 减少 DRAM 搬运量的三种手段**（融合 / 降精度 / 不写无用输出） —— 产出：`vecadd_fused.cu` 对照实验 + 报告 · PG §5.3

### D. 语言扩展（常用的那一批）

- [x] **D1 执行空间与内存空间限定符** —— `cuda/vecadd.cu`（`__global__` / `const __restrict__`）· PG §7.1–7.2
- [x] **D2 内建向量类型** —— `cuda/vecadd.cu`（`float4` / `make_float4`）· PG §7.3
- [ ] **D3 同步与内存栅栏**（`__syncthreads` / `__threadfence`） —— PG §7.5–7.6
- [ ] **D4 warp 级原语**（vote / shuffle / reduce） —— 产出：用 shuffle 写一个 warp 内求和 · PG §7.19–7.22
- [ ] **D5 原子操作** —— 产出：用 atomicAdd 做直方图或计数器 · PG §7.14
- [ ] **D6 流与异步执行**（streams / events / 重叠） —— 产出：两条流并发的对照实验 · PG §3.2
- [ ] **D7 启动配置与寄存器上限**（`__launch_bounds__` / `-maxrregcount`） —— PG §7.37–7.39
- [ ] **D8 `#pragma unroll` 与编译器优化提示** —— PG §7.18, §7.40
- [ ] **D9 异步拷贝的硬件边界** —— **证据已齐**（`-arch=sm_75` 生成 0 次 `cp.async`、sm_80 生成 3 次的对照输出），**待复述后打勾** · 知道本机没有 `cp.async`（sm_80+）、没有 TMA · PG §7.27–7.29
- [ ] **D10 纹理 / 常量内存路径** —— PG §7.8, §7.10

### E. 工具链

- [x] **E1 nvcc 编译流程与常用开关** —— `cuda/build.ps1`, `notes/03` · 能说出 `-arch` / `-Xptxas -v` / `-cubin` 各自干什么 · PG §3.1
- [x] **E2 PTX / SASS / cubin 三层关系** —— `notes/06`, `cuda/reg_demo.ptx` · CUDA Binary Utilities
- [x] **E3 用 cuobjdump / nvdisasm 读汇编** —— `notes/06`, `reports/sector_probe.txt` · CUDA Binary Utilities
- [x] **E4 ncu 指标体系与定义查询** —— `notes/06`, `notes/05` · 能自己从 `sections/*.section` 查出指标定义 · NCU Profiling Guide
- [x] **E5 结论必须配原始证据的规范** —— `AGENTS.md`, `reports/` · —
- [ ] **E6 计算卫生检查**（compute-sanitizer / 越界与竞态） —— 产出：用 compute-sanitizer 跑一遍现有 kernel

### F. 实战产出

- [x] **F1 第一个完整 kernel 的全流程拆解** —— `notes/04` · —
- [x] **F2 memory-bound 三问三答** —— `reports/membound_qa.md` · —
- [x] **F3 硬件标定 + roofline 图** —— `calibrate.py`, `hardware.json`, `roofline.png` · —
- [ ] **F4 融合 vs 多遍遍历对照实验** —— 见 C10
- [ ] **F5 fp16 存储的流量收益实验** —— 见 C10

---

## 3. L2–L5 的边界（现在只需要知道"什么时候展开"）

这些关卡**现在不展开**，避免一次给出几十个名词。展开条件写死如下：

| 关卡 | 主题 | 展开条件 | 展开时会先给什么 |
|---|---|---|---|
| L2 | Kernel 工程化：共享内存、tiling、指令吞吐、warp 原语 | L1 勾满 45/45 | 一份 12 条的 L2 清单 |
| L3 | 并行训练与通信：NCCL、集合通信、ZeRO/FSDP、流水线并行 | L2 勾满 **且** 有了多卡环境（本机单卡，需上云或换机） | 清单 + 环境需求 |
| L4 | 推理系统：KV cache、batching、量化、调度 | L3 完成 | 清单 |
| L5 | 编译与图优化、系统规模化 | L4 完成 | 清单 |

> 这份路线图只承诺两件事：**L1 的 34 条是完整的**；后面每一关的**展开条件**是明确的。
> 所以"什么时候能学完"的答案始终具体：**看当前关卡还剩几条。**

---

## 4. 名词表（防止"突然冒出新名词"）

规则：本清单之外出现的任何概念，都必须先登记到这里，标注是**计划内**还是**计划外补充**，
由你决定是否加进 §2。现有名词登记位置：

| 已有登记 | 位置 |
|---|---|
| 硬件执行单元、存储层级、性能概念 | `notes/03 §八 术语总表` |
| 编译/程序/硬件/存储四类名词 | `notes/06 §8 名词对照` |
| MSVC / 路径 / 编码等踩坑 | `README.md` 「其他坑」 |

计划外补充（本轮之前的对话里新增的，需你确认是否并入 §2）：

| 名词 | 出现位置 | 建议 |
|---|---|---|
| sector / cache line / memory transaction | 已并入 B5 | 已入库 |
| L2 write-allocate without fetch、write-back | 已并入 B6 | 已入库 |
| PTX 缓存提示（`.cg/.cs/.wt`）、`prefetch`、`discard` | 已并入 B7 | 已入库 |
| `cp.async` / TMA / L2 驻留控制 | 已并入 D9 | 已入库 |
| 融合、降精度、DRAM 搬运量 | 已并入 C10 | 已入库 |

**2026-10-03 新增（来自 `notes/09`，是我的流程失误：写的时候没先登记）**：

| 名词 | 一句话 | 建议归属 | 是否并入 §2 |
|---|---|---|---|
| CUDA Graph | 把一串 kernel 启动录成一张图、一次提交，把 N 次启动开销压成 1 次 | 新增 **C11 减少启动开销**（约 30 分钟可做实验） | 待定 |
| 持久化 kernel | 常驻 kernel（block 数 = 全机容量）循环领任务，永不退出 | 同上 | 待定 |
| 软件流水 | 把"读下一块"与"算当前块"错开，用计算盖住访存延迟 | 归入 **C7（指令吞吐）** 或新增延迟隐藏条 | 待定 |
| ILP / MLP | 指令级 / 内存级并行：一个线程挂多笔独立访存 | 归入 **C5 / C7** | 待定 |

> 这四个都**还没有实测证据**，也不在你已经打勾的 27 条里。
> 按 §0 规矩 2，加不加进 §2 由你决定——**不加的话，我以后就不在对话里展开它们**。

---

## 5. 出处（清单不是编的）

本清单的 A/B/C/D 分组直接对应官方文档的章节结构：

| 官方资料 | 本清单用到的部分 | 地址 |
|---|---|---|
| CUDA C++ Programming Guide | A←ch.2、B←ch.4、C←ch.5、D←ch.7、E←ch.3 | https://docs.nvidia.com/cuda/cuda-programming-guide/ |
| CUDA C++ Best Practices Guide | C1–C6 的优化优先级 | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ |
| Turing Tuning Guide | 本机 sm_75 的专属建议（调优时查） | https://docs.nvidia.com/cuda/turing-tuning-guide/ |
| PTX ISA | B7、D9 的指令级细节 | https://docs.nvidia.com/cuda/parallel-thread-execution/ |
| Nsight Compute Profiling Guide | E4 的指标定义 | https://docs.nvidia.com/nsight-compute/ProfilingGuide/ |

本机限制（决定了哪些条目"只需知道边界"）：无 Tensor Core、无 `cp.async`、无 TMA、
无 L2 驻留控制、显存 4 GB。详见 `README.md` 约束一、`notes/06 §4`。

---

## 6. 节奏建议

| 项 | 建议 |
|---|---|
| 每条耗时 | 30–90 分钟（含写产出物） |
| 每周 | 1–2 条；宁可慢，必须留证据 |
| 卡住时 | 先写"卡在哪一步 + 已经排除了什么"，再问 |
| 每次提问 | 我会先说明这个问题属于清单里的哪一条；如果是清单外的，先登记再讲 |
