# AI Infra 学习

按阶段推进。每个阶段必须留下一个可复现的产出物，没有产出物就等于没学。

---

## 环境结论（实测，2026-09-30）

| 项 | 值 |
|---|---|
| GPU | NVIDIA T600 Laptop，sm_75 (Turing)，14 SM / 896 FP32 核 / 4.29 GB |
| 驱动 / toolkit | 592.82 (支持 CUDA 13.1) / 本地 nvcc 12.6 |
| PyTorch | 2.6.0+cu124，CUDA 可用 |
| Triton | 3.8.0（triton-windows），已在 sm_75 上跑通 |
| Nsight Compute | 2024.3.1 已装，但需管理员权限 |
| MSVC | VS Build Tools 17.14（MSVC 14.44）已安装，nvcc 可用 |
| WSL2 | Ubuntu 26.04，GPU 直通正常，但无 toolkit / 无 gcc / Python 3.14 装不上 torch |

### 实测性能基线

由 `phase0/calibrate.py` 测出，完整数据在 `phase0/hardware.json`。

- 显存带宽：copy 172 / read 179 / write 185 GB/s
- fp32 峰值：**2.76 TFLOP/s**
- fp16：0.32 TFLOP/s（比 fp32 还慢，见下）
- Roofline 转折点：**15.4 FLOP/byte**

### 三个必须记住的约束

**1. 没有 Tensor Core。** T600 Laptop 用的是 TU117，和 GTX 16 系同源，是 Turing
家族里砍掉 Tensor Core 的那一支。表现就是 fp16 只有 fp32 的 0.12 倍，完全倒挂。
后果：`mma` / `wgmma` / FlashAttention / fp8 量化这些内容**本机练不了**，
必须上云（一张 T4 就有 TC，按小时租很便宜）或换卡。

**2. nvcc 依赖 MSVC，且项目路径含中文会炸。**
  VS 2022 曾经装过（卸载列表里留着 NVIDIA CUDA VS Integration 和 Nsight VS Edition
  的痕迹），后来被卸载了，现已重装 VS Build Tools 17.14。

  编译 .cu 请统一走 `phase0/cuda/build.ps1`，因为它替你绕开了两个坑：
  - `cl.exe` 不在默认 PATH，必须先执行 `vcvars64.bat`
  - 项目路径含中文，`cmd.exe` 默认 GBK 代码页，传绝对路径会变乱码
    （表现为 `LNK1104: 无法打开文件`）。脚本用相对路径 + `chcp 65001` 解决。

  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File phase0\cuda\build.ps1 phase0\cuda\hello.cu
  ```

**3. ncu 需要管理员权限。** 直接跑会报 `ERR_NVGPUCTRPERM`。
  用管理员身份打开 PowerShell 再跑即可：

```powershell
ncu --set full --launch-count 1 --launch-skip 2 python phase0/xxx.py
```

### 其他坑（已踩）

- pip 直连 pypi 会卡死（半小时 0 字节）。用清华镜像：
  `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <pkg>`
- triton-windows 3.8 和 torch 2.6 不兼容，`torch.compile` 会报
  `cannot import name 'AttrsDescriptor'`。手写 Triton kernel 不受影响，
  到阶段 4 再处理（降 triton 或升 torch）。
- **`.cu` / `.cuh` 必须存成带 BOM 的 UTF-8。** 否则 MSVC 按 GBK(936) 解析，
  中文注释会把代码结构搞坏，报出莫名其妙的语法错误。
  `build.ps1` 已经加了 `-Xcompiler /utf-8`，但源码带 BOM 才是根治。
- **nvcc 处理非 ASCII 字符串字面量完全不可靠**（已三次踩坑）：有时能编译，
  有时报 `missing closing quote`，有时悄悄吃掉紧挨中文的转义符（`\n` 变成字面量）。
  **硬规则：`.cu` / `.cuh` 里的字符串字面量一律只用 ASCII，中文只进注释。**
- PowerShell 5.1 没有 `[IO.Path]::GetRelativePath`（那是 .NET Core API），
  PS 脚本里要用字符串截断代替。

---

## 进度

- [x] 阶段 0 · 度量能力 ← 完成
- [ ] 阶段 1 · Kernel 工程化
- [ ] 阶段 2 · 并行训练与通信
- [ ] 阶段 3 · 推理系统
- [ ] 阶段 4 · 编译与图优化
- [ ] 阶段 5 · 系统与规模化

## 阶段 0 · 度量能力

目标：把"快了"从感觉变成数字。之后所有优化的分母都来自这一阶段。

- [x] 环境探测
- [x] 计时工具 `phase0/bench.py`
- [x] 硬件标定 `phase0/calibrate.py` → `phase0/hardware.json`
- [x] roofline 图 `phase0/roofline.py` → `phase0/reports/roofline.png`
- [x] 第一个 kernel + ncu 报告 → `phase0/reports/vecadd_profile.md`

### 结论速览

向量加实测 179.9 GB/s，DRAM 利用率 93.31%，SM 利用率 4.98%。
贴着带宽屋顶，**零优化空间** —— 向量化和 block size 扫描都只带来 1% 以内的差异。
这是 memory-bound kernel 的教科书样本，也定义了阶段 0 的核心技能：
**先判断 kernel 卡在算力还是带宽，再决定优化方向。**

进一步做了三个对照实验（向量宽度 / block size / 并行度下限），
结论见 `phase0/reports/membound_qa.md`：连 float3 这种指令数翻三倍的写法
都不变慢，说明瓶颈资源饱和后，其余所有优化都是零收益。

还做了一个 `volatile` 对照组（`reg_reuse.cu`）：同一份代码去掉 `volatile`，
SASS 里两条 `LDG` 立刻被提到循环外，小工作集档快 **11~13 倍**——
而 DRAM 字节数**分毫不变**。结论见 `phase0/reports/reg_reuse_qa.md`：
**寄存器复用是编译器免费送的，缓存只在它证不出来时才接管。**

### 用法

```powershell
# 标定硬件（约 20 秒）
python phase0/calibrate.py

# 画 roofline（读 hardware.json + measured/*.json）
python phase0/roofline.py

# 编译并运行 CUDA kernel
powershell -NoProfile -ExecutionPolicy Bypass -File phase0\cuda\build.ps1 phase0\cuda\vecadd.cu
.\phase0\cuda\vecadd.exe

# 用 ncu 采计数器（需要管理员权限）
ncu --set full -k "regex:vecadd_float4" --launch-skip 20 --launch-count 1 .\phase0\cuda\vecadd.exe
```

`do_bench` 的 `warmup` / `rep` 单位是**毫秒**而不是次数，内部会先估单次耗时再反推轮数。
默认每轮之间刷 L2，避免测出 cache 速度冒充 DRAM 速度。
---

## 笔记索引

`phase0/notes/` 下的九篇笔记是概念地基，随时回来翻；另有一篇可视化页
`10_gemm_tiling_visual.html`（浏览器直接打开，无需联网）。

> 学习总纲在 [`LEARNING_PATH.md`](LEARNING_PATH.md)：先看范围、再看进度，新名词先登记再学。

| 文件 | 讲什么 | 什么时候看 |
|---|---|---|
| `01_memory_basics.md` | 访存、SM 利用率、合并访存、在途请求 | 看不懂某个性能数字时 |
| `02_hardware_arch.md` | 硬件层级、warp、寄存器预算、硬件与 CUDA 模型的区别 | 要讲清楚"为什么慢"时 |
| `03_factory_analogy.md` | 用"工厂"比喻把上面全部重讲一遍 + 术语中英对照 + 常见误解 + ncu 速查卡 | **卡住时先看这篇** |
| `04_vecadd_walkthrough.md` | 向量加的完整执行流程：从一行 CUDA 到 448 个班组轮 4681 次班 | **想按时间顺序过一遍全流程时** |
| `05_roofline_model.md` | roofline 的三步推导、为什么是硬上界、逐元素读图、四个盲区 | **要判断"该优化算力还是带宽"时** |
| `06_sector_and_instructions.md` | sector 是什么、SASS 实测（float3 为什么 3 条指令也不慢）、资料地图 | **想知道该查哪些资料、为什么要看汇编时** |
| `07_performance_factors.md` | 性能因素总表：五层决策顺序 + 13 个因素（含"什么时候不用管"）+ 混淆点 | **动手优化任何 kernel 之前** |
| `08_ncu_map.md` | ncu 使用地图：2344 个指标怎么按单元/问题找，常用命令与三个坑 | **要自己查指标、自己采数时** |
| `09_kernel_optimization_workflow.md` | 拿一个 kernel 该怎么做：五步循环 + 按瓶颈查杠杆表 + 收敛判据 | **每次接到新 kernel 时，第一步先看这篇** |
| `10_gemm_tiling_visual.html` | SGEMM 四版优化全流程图解（换位置 → 换次数 → 换方式，含合并访存、瓶颈搬家、BLAS 命名法） | **看不懂 tiling 为什么成立时** |

配套的可执行依据：

| 文件 | 作用 |
|---|---|
| `phase0/cuda/device_query.cu` | 打印本机全部硬件参数，是所有性能讨论的依据 |
| `phase0/hardware.json` | 实测的带宽/算力峰值，roofline 的分母 |
| `phase0/reports/vecadd_ncu.txt` | 完整的 ncu 计数器原始输出 |
| `phase0/reports/membound_sweep.txt` | 向量宽度 / block / 并行度三个扫描的原始数据 |
| `phase0/reports/sector_probe.txt` | SASS 与 sector 计数器的原始输出（笔记 06 的证据） |
| `phase0/reports/cache_reuse_qa.md` | 数据量 / 复用距离 / 启动开销的实测报告（含对原预测的勘误） |
| `phase0/reports/cache_probe_ncu.txt` | `cache_probe` 的原始输出（独立运行 + ncu 逐档 + 官方 roofline section） |
| `phase0/cuda/reg_reuse.cu` | `volatile` 对照组：同一份代码去掉 `volatile`，看编译器会不会把访存吃掉 |
| `phase0/reports/reg_reuse_qa.md` | `volatile` 对照组的实测报告（SASS + 计数器 + 耗时三方对账） |
| `phase0/reports/reg_reuse_sass.txt` | 两个 kernel 的 SASS 原始输出（`LDG` 到底在不在循环里） |
| `phase0/reports/reg_reuse_ncu.txt` | `reg_reuse` 两档 x 两 kernel 的 ncu 计数器原始输出 |
| `phase0/cuda/sgemm.cu` | SGEMM 四版：naive / tiled / 寄存器分块 / reg2（对应笔记 10） |
| `phase0/reports/sgemm_qa.md` | SGEMM 四版的实测报告（瓶颈搬家 + 官方 rules 复现 + 两个测量口径的坑） |
| `phase0/reports/sgemm_ncu.txt` | `sgemm` 四版的 ncu 计数器原始输出 |
| `phase0/reports/sgemm_reg_rules.md` | `sgemm_reg` 的 ncu 官方规则报告（10 条，5 条带收益估计） |
| `phase0/reports/sgemm_reg2_rules.md` | `sgemm_reg2` 的 ncu 官方规则报告（改完之后，瓶颈换了新目标） |
| `phase0/ncu_rules.py` | 从 `ncu --set full` 输出里抽出 OPT 规则表的小工具 |
| `phase0/cuda/reg_demo.ptx` | 一份手写 PTX（笔记 06 / E2 的证据，说明 PTX 长什么样） |
