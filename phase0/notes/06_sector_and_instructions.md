# 笔记 06 · sector、访存指令与资料地图

前置：`01_memory_basics.md`（合并访存）、`03_factory_analogy.md`（工厂）、
`05_roofline_model.md`（盲区 2）。

原始证据：**`phase0/reports/sector_probe.txt`** —— 本文出现的每一个计数器、
每一条 SASS、每条命令都在那个文件里。

---

## 0. 一句话

> **钱按 sector 付，指令只是下单方式。**
> 同一份数据用 1 条还是 9 条指令去取，DRAM 的账一分不差；
> 变的只是"经手次数"。

---

## 1. sector 是什么

Nsight Compute Profiling Guide 术语表的原文（这两句是整个阶段 0 的地基）：

> **Sector**: Aligned 32 byte-chunk of memory in a cache line or device memory.
> **An L1 or L2 cache line is four sectors, i.e. 128 bytes.**
> Sector accesses are classified as hits if the tag is present and the sector-data
> is present within the cache line.

> Total number of bytes requested from L1. This is identical to the number of sectors
> multiplied by 32 byte, **since the minimum access size in L1 is one sector**.

```text
sector      = 32 B    <- 最小搬运 / 记账单位（L1、L2、DRAM 共用）
cache line  = 128 B   = 4 个 sector
一次内存事务  >= 1 个 sector
```

| 层级 | 本机容量 | 最小访问单位 | 谁决定 |
|---|---|---|---|
| 寄存器 | 256 KB / SM | 4 B（1 格） | 编译器 |
| L1 / 共享内存 | 96 KB / SM | **1 sector = 32 B** | 硬件 |
| L2 | 1 MB | **1 sector = 32 B** | 硬件 |
| DRAM | 4 GB | **1 sector = 32 B** | 硬件 |

**所以"内存事务"和 sector 是同一件事的两面**：说"合并访存"的精确表述是——
**让一个 warp 的 32 个线程要的数据，落在尽可能少的 sector 里**。

### 理想值公式

```text
sectors / 请求（理想） = 每线程字节数
```

因为一个 warp 是 32 个线程 × B 字节 = 32B 个字节，除以 32 B/sector 正好是 B：

| 每线程访问 | 一个 warp 要的字节 | 理想 sectors/请求 |
|---|---|---|
| 4 B（float） | 128 B | 4 |
| 8 B（float2 / double） | 256 B | 8 |
| 12 B（float3） | 384 B | 12 |
| 16 B（float4） | 512 B | 16 |

### 单位换算：千万别把 sector 当成"多少个 float"

这是最容易卡住的一步。三个单位的关系：

| 单位 | 大小 | 备注 |
|---|---|---|
| 1 个 float32 | 4 B | |
| **1 个 sector** | **32 B = 8 个 float32** | 记账单位 |
| 一次 warp 请求 | 32 个线程各自的访问 | 指标的分母 |

**关键等式（比记公式好用）：**

```text
等效每线程字节数 = sectors/请求 ÷ 32 × 32 = sectors/请求
理想值           = 实际每线程字节数

所以：sectors/请求 > 每线程字节数  ->  多出来的就是浪费
```

换句话说：**`sectors/请求` 的数值，直接就是"系统认为你每个线程占用了多少字节"**。

举例（用本机三个 kernel 验证，全部精确吻合）：

| kernel | 实际每线程字节 | 实测 sectors/请求 | 判定 |
|---|---|---|---|
| `vecadd_naive` | 4 | 4 | 无浪费 |
| `add_f3`（float3） | 12 | 12 | 无浪费 |
| `vecadd_float4` | 16 | 16 | 无浪费 |

### 反例：`sectors/请求 = 24` 而每线程只有 16 字节，长什么样

每线程读 16 B（一个 `float4`），但**步长是 24 B**（比如数组元素被 padding 到 24 字节，
或者从 AoS 结构里取一列）：

```text
线程 0: 字节 [0, 16)      线程 1: [24, 40)     线程 2: [48, 64)  ...
整个 warp 跨 [0, 760)  ->  覆盖 sector 0..23  ->  24 个 sector

需要：32 线程 x 16 B = 512 B
实取：24 sector x 32 B = 768 B
浪费：1.5 倍（多搬 50%）
```

因为 16 字节的访问不落在 32 字节对齐的格子里，**相邻线程的访问被错开**，
同一批 sector 被反复进出、边缘 sector 只用到一半。

**修法**：把步长改回等于访问宽度（SoA 布局、去掉 padding、或对齐到 32 B），
浪费立刻从 1.5 倍降到 1.0 倍——**这一项比"减少算法搬运量"往往来得更快**。

### 防混淆：本仓库里所有的 "32" 和 "128"

同一批笔记里既有"寄存器一格 = 32 位"，又有"一个 sector = 32 字节"，
两个 32 撞车、量级差 8 倍，极易记串。一次列清：

| 说法 | 单位 | 换算 | 出处 |
|---|---|---|---|
| 寄存器**一格** | 32 **位** | = 4 字节 | 笔记 03「一个格子有多宽」 |
| 一条 `LDG.E`（无 `.64/.128` 尾缀） | 32 **位** | = 每线程读 4 字节 | 本文 §4 |
| 一个 **warp** | 32 **线程** | — | 笔记 03 §四 |
| **一个 sector** | 32 **字节** | = 256 位 = 8 个 float32 | 本文 §1 |
| 一个 **cache line** | **128 字节** | = 4 个 sector | 本文 §1 |
| 每 SM 上限 | 32 个 warp / 32 个 block | — | 笔记 02 |
| 内存**总线宽度** | **128 位** | = 16 字节 | `device_query.txt` |

> **规律："位"只出现在寄存器和总线宽度上；"字节"才出现在 sector 和 cache line 上。**
> 两条 128 也是完全无关的两件事：总线 128 位 = 16 字节，cache line 128 字节。

单位约定：`B` = Byte（字节）、`b` = bit（位），**1 B = 8 b**。
`179 GB/s` 和 `192 GB/s` 都是字节；`128-bit 总线` 是位。

**交叉验证（一条官方描述 + 一条实测）**：NCU 文档说
"Each request accesses **up to four sectors from a single 128 byte cache line**"。
`vecadd_float4` 一个 warp 要 512 B → 拆成 4 个 cache line × 每个 4 sector = **16 sectors**，
与实测的 `sectors/请求 = 16` 精确吻合。

### 常见追问：sector 是"内存事务的最小单位"吗？最少要搬 8 个 float32 吗？

**第一问：是，但要分清是哪一段的。** 一条访存取数经过三层，粒度不同：

```text
线程发出请求      LDG.E，可以只要 4 字节            <- 这是"请求"，不是"事务"
      ↓
L1 / L2 处理      按 sector 处理，最小 32 字节       <- "the minimum access size in L1 is one sector"
      ↓
L2 <-> DRAM      同一粒度，32 字节                   <- 一次事务的最小量
```

所以 **"一次内存事务最小 = 1 个 sector = 32 字节"成立**。
但注意区分：线程**可以只请求 4 字节**，只是这一请求会把整个 32 字节的 sector 拉进来。

**第二问：方向反了。** 关键在"8 个 float32 是**每次事务**的粒度，不是**每个线程**的代价"：

| 场景 | 一次事务搬多少 | 其中被用上的 | 浪费 |
|---|---|---|---|
| 32 个线程读连续的 float（每线程 4 B） | 4 个 sector = 128 B = **32 个 float32** | 32 个（每线程 1 个） | **0** |
| 32 个线程读 `a[i*8]`（线程间步长 32 B） | **32 个 sector** = 1024 B = **256 个 float32** | 32 个（每线程 1 个） | **7/8 = 87.5 %**（相对理想 4 个 sector 是 **8 倍**） |

对照本机实测：`vecadd_naive` 的 `sectors/请求 = 4`，
正好等于 "32 线程 × 4 字节 ÷ 32"——**8 个 float32 的粒度被完全用满，一分钱没浪费**。
只有当访问模式不配合（跨步、只取一列、warp 发散）时，那 8 个里才只用到 1 个。

> **一句话：sector 决定"账怎么记"，合并访存决定"账有没有被摊薄"。**
> 32 字节粒度不是让你吃亏，而是为了把相邻线程的零散请求拼成一车货。

### 术语：stride（步长）到底指什么

**步长 = 相邻两次访问之间的地址间隔。** GPU 里要紧的是前一种：

| 种类 | 定义 | 例子 | 影响什么 |
|---|---|---|---|
| **线程间步长** | 相邻**线程**访问的地址间隔 | 线程 i 读 `a[i*8]` → 间隔 8 个 float = 32 B | **能否合并访存**（决定 sector 数） |
| 线程内步长 | **同一个线程**循环里两次访问的间隔 | `for (k = 0; k < N; k += 32) sum += a[k];` | 缓存局部性、能否向量化 |

**`a[i*8]` 长什么样**（8 个 float = 32 字节，正好一个 sector）：

```text
线程号    0    1    2    3   ...   31
字节地址  0   32   64   96   ...  992
落在      0    1    2    3   ...   31   <- 每个线程独占一个 sector
用到      4B   4B   4B   4B         <- 每 sector 只用 4/32 = 12.5%
```

一次请求触及 **32 个 sector**，理想只需 **4 个**（128 B）→ **多搬 8 倍**，
其中 87.5% 是白搬的。（"浪费 7/8"和"多搬 8 倍"是同一件事的两种说法：
前者是**没用到**的比例，后者是**开销**的倍数。）

**真实代码里它长这样**（不同写法，同一个病）：

- 结构体数组（AoS）只取一个字段：`struct P { float x, y, z; float pad; }; ... p[i].x`
- 二维数组按**列**访问，但线程沿**行**展开
- 矩阵乘法里 A 按行读、B 按列读（转置一下就好）
- 索引算错：本意是 `a[i]`，写成了 `a[i * 8]`

**修法**：让相邻线程访问相邻地址——SoA 布局、转置、分块（tiling）。

---

## 2. 实测账本：vecadd

| kernel | 每线程字节 | sectors/请求（实测） | L1 ld sector 总数 | L1 st sector 总数 |
|---|---|---|---|---|
| `vecadd_naive` | 4 | **4** | 16,777,216 | 8,388,608 |
| `vecadd_float4` | 16 | **16** | 16,777,216 | 8,388,608 |

两个版本的总数**一模一样**，而且正好等于义务流量：

```text
16,777,216 * 32 B = 512 MiB   （读 a + 读 b）
 8,388,608 * 32 B = 256 MiB   （写 c）
```

**一个 sector 都没多搬** —— 这就是"完美合并访存"在硬件账本上的样子。

### 可复用的自查判据

```text
L1 sector 总数 * 32 == 实际搬运字节数        -> 合并完美
左边更大                                     -> 多出来的是白搬的（未对齐/跨步/发散）
```

另有 `dram__bytes_read` / `dram__bytes_write` 直接给出 DRAM 侧的真实字节数，
和 L1 请求数对照就能看出"浪费发生在哪一级"。

### "有效带宽"是什么：多级存储共同作用的加权平均

平时说的"带宽 179 GB/s"是**某一级**的物理峰值。而实验里算出来的那个数，
通常是这样得到的：

```text
有效带宽 = 程序请求的总字节数 / 耗时
```

**分子数的是"程序要了多少"，不管数据从哪一级来**——所以它是 L1 / L2 / DRAM
共同作用的结果。但**不是算术平均**，因为时间相加、速率不能相加：

```text
有效带宽 = 1 / ( f_L1/R_L1 + f_L2/R_L2 + f_DRAM/R_DRAM )
           where f = 各级服务掉的流量占比，R = 各级的速率

哪一级的占比大，结果就由那一级主导。
```

`cache_probe` 的两档正好演示这一点（扣掉固定开销后的有效带宽）：

| 配置 | L2 命中 | 由谁主导 | 有效带宽 |
|---|---|---|---|
| k <= 8 | 97 ~ 98 % | **L2** | **282 GB/s** |
| k >= 12 | 0.5 ~ 0.9 % | **DRAM** | **180 GB/s** |

**不是"两个东西在变"，而是权重在变。**

> 注意本实验里 **L1 完全不参与**——L1 命中率实测 0 %（`volatile` 让读绕过 L1），
> 所以是"L2 + DRAM 两级"，不是三级。

**实践含义**：看到"有效带宽"这个数，**必须配合命中率一起看**；
单独一个数字无法告诉你瓶颈在哪一级。这也正是 roofline 需要"分层"的原因
（笔记 05 盲区 2）——roofline 的 Y 轴画的就是有效带宽。

### 为什么请求数翻 4 倍，带宽还能打满

实测（原始输出见 `reports/sector_probe.txt` 第 [5] 节）：

| 指标 | `vecadd_naive`（float） | `vecadd_float4` | 差异 |
|---|---|---|---|
| L1 ld **请求数** | **4,194,304** | **1,048,576** | **4 倍** |
| sectors/请求 | 4 | 16 | 4 倍 |
| L1 ld **sector 总数** | 16,777,216 | 16,777,216 | **完全相同** |
| L1 流水线吞吐 | 26.30 % | 26.34 % | 相同 |
| DRAM 吞吐 | 93.40 % | 93.66 % | 相同 |
| 耗时 | 4.49 ms | 4.48 ms | 相同 |

原因是三个环节各看各的账：

| 环节 | 它关心什么 | 两个 kernel 的差别 | 该环节忙不忙 |
|---|---|---|---|
| SM 发射（指令侧） | **请求数**、指令数 | **4 倍** | 只用 5 %，闲 |
| L1 / L2 | **sector 总数** | 完全相同 | 26 %，闲 |
| DRAM | **字节数** | 完全相同 | **93 %，满** |

**请求数只压在"发射"那一环，而那一环本来空着 95 %。**
带宽不是被"请求数"喂饱的，是被**在途字节数**喂饱的：

```text
带宽 = 在途字节数 / 延迟                （Little's law）
打满 179 GB/s 需要在途约 107 KB         （笔记 03 第七节算过）

float4：107 KB / 512 B = 209 个请求在途
float ：107 KB / 128 B = 838 个请求在途   <- 需要 4 倍的车，但车队够大
```

**工厂比喻**：装卸口一次能过 32 件货。
- 大卡车（float4）一趟 16 件 → 2 趟装满
- 小卡车（float）一趟 4 件 → 8 趟装满

只要**同时在路上跑的车够多**，装卸口就不会空转。总吞吐由装卸口决定，与车的大小无关。

**什么时候这条会失效**：车队不够用时——即请求数把"发射侧"或"在途请求槽位"填满时。
典型场景是占用率低、SM 利用率高、对延迟敏感。**那才是"请求数变多开始收税"的时刻。**

### 三个量别混：请求数 / sector 数 / L1 吞吐

```text
请求数          = (线程数 / 32) x 每线程的访存指令数
sectors/请求    = 由"每线程字节数 + 地址分布 + 对齐"决定
sector 总数     = 请求数 x sectors/请求        <- 这才是 DRAM 账本的分子
L1 吞吐         = 由 sector 总数决定，不由请求数决定
```

实测对照（vecadd 两个版本，数据量都是 512 MiB）：

| 量 | `vecadd_naive`（float） | `vecadd_float4` |
|---|---|---|
| 线程数 / warp 数 | 67.1M / 2,097,152 | 16.8M / 524,288 |
| 每 warp 的访存指令 | 2（读 a、读 b） | 2 |
| **请求数** | **4,194,304** | **1,048,576**（4 倍差） |
| sectors/请求 | 4 | 16 |
| **sector 总数** | **16,777,216** | **16,777,216**（相同） |
| **L1 吞吐 %** | **26.30 %** | **26.34 %**（相同） |

两句话记住：

- **请求数跟"有多少 warp × 每条 warp 几条访存指令"有关**（所以确实和并行度沾边），
  但它**不决定 L1 有多忙**。
- **真正决定 L1 忙不忙的是 sector 总数**。判断"是不是被 L1 拖累"，
  要看 `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`，不是看请求数。

> **更正一个常见直觉**："每个请求的大小由 L1 / 寄存器容量决定" —— 不对。
> 请求大小（sectors/请求）只由**访问模式**决定（每线程字节数、地址分布、对齐）。
> L1 / 寄存器的**容量**决定的是另一件事：**能同时驻留多少 warp**（占用率），
> 也就是"能有多少请求同时在途"。

---

## 3. `L2 Hit Rate 33.33%` 的真相

| 查询类型 | sector 数 | 命中 | 命中率 |
|---|---|---|---|
| 读 a / b | 16,794,312 | 5,313 | 0.03 % |
| 写 c | 8,390,349 | 8,388,672 | ~100 % |
| 合计 | 25,184,661 | 8,393,985 | **33.33 %** |

DRAM 侧：读 536.91 MB（512 MiB）、写 268.36 MB（256 MiB），合计正好是义务流量。

三个要点：

1. **写查询算命中**：写满整个 sector 的 store 不需要先把数据从 DRAM 取回来
   （write-allocate without fetch），所以它的 lookup miss = 0。
2. **33% 是读写比例的镜像**：读写 2:1，命中全来自那 1 份写，所以 1/3。
3. **它和"数据复用"没关系**。判断复用要看 `lts__t_sectors_op_read_lookup_hit`；
   这里只有 0.03%，那还是 ncu 多次 replay 的残留。

---

## 4. 从 SASS 验证"float3 是 3 条指令"

```powershell
nvcc -arch=sm_75 -cubin -Xptxas -v -o <tmp>\membound_sweep.cubin phase0\cuda\membound_sweep.cu
cuobjdump -sass <tmp>\membound_sweep.cubin
nvdisasm -c <tmp>\membound_sweep.cubin        # 同上，换一个工具
```

实测 SASS 摘录（完整清单见 `reports/sector_probe.txt`）：

```text
add_f1   LDG.E.CONSTANT.SYS R4, [R4]        LDG.E.CONSTANT.SYS R3, [R2]
         STG.E.SYS [R6], R9

add_f3   LDG.E.CONSTANT.SYS R0, [R4]        LDG.E.CONSTANT.SYS R8,  [R4+0x4]
         LDG.E.CONSTANT.SYS R10,[R4+0x8]    LDG.E.CONSTANT.SYS R7,  [R2]
         LDG.E.CONSTANT.SYS R11,[R2+0x4]    LDG.E.CONSTANT.SYS R13, [R2+0x8]
         STG.E.SYS [R6], R9                 STG.E.SYS [R6+0x4], R11
         STG.E.SYS [R6+0x8], R13

add_f4   LDG.E.128.CONSTANT.SYS R8, [R2]    LDG.E.128.CONSTANT.SYS R12, [R4]
         STG.E.128.SYS [R6], R8
```

| kernel | 访存指令 | 明细 | 总指令数 |
|---|---|---|---|
| `add_f1` | 3 | 2 × `LDG.E` + 1 × `STG.E` | 16 |
| `add_f2` | 3 | 2 × `LDG.E.64` + 1 × `STG.E.64` | 24 |
| `add_f3` | **9** | 6 × `LDG.E` + 3 × `STG.E` | 24 |
| `add_f4` | 3 | 2 × `LDG.E.128` + 1 × `STG.E.128` | 24 |

两个值得记住的细节：

- **12 字节没有对应的位宽**。硬件只有 32 / 64 / 128 位三种箱子，所以 `float3` 被拆成
  3 条 32 位，地址用立即数偏移 `[R4+0x4]`、`[R4+0x8]` 表达。
  **不是编译器偷懒，是硬件没这个规格。**
- **`.CONSTANT` 来自 `const __restrict__`**（走只读数据路径）。指令名里的修饰符会随写法变，
  这也是"指令集不能靠背"的直接例证。

### 为什么 3 倍指令也不慢

`add_f3` 的实测（每线程 12 B，全部变体总字节数恒为 720 MB）：

| 指标 | 值 | 说明 |
|---|---|---|
| L1 ld sector 请求 | 47,185,920 | = **3 ×** 15,728,640（1 份本该有的量） |
| L1 st sector 请求 | 23,592,960 | = **3 ×** 7,864,320 |
| sectors/请求 | **12** | = 每线程 12 B ✓ 命中理想值 |
| **L1 命中率** | **66.65 %** | 后两条加载要的数据，正是第一条加载进来的 sector |
| DRAM 读 / 写 | 503.34 / 251.68 MB | = 480 / 240 MiB，**义务流量，一分没多** |

推理链：

1. 每条 `LDG.E` 的 32 个地址按 12 字节步长展开，跨 384 B = **12 个 sector**；
2. 第 1 条把这些 sector 拉进 L1（miss），后两条落进同一批 sector（hit）
   -> 命中率 `2/3 = 66.65 %`，正好对上；
3. 所以 **DRAM 只搬 1 份**，多出来的 2 份是 **L1 内部的请求**。

结论：**`float3` 多花的是"L1 请求次数"和"发射槽位"，不是"DRAM 搬运量"。**
vecadd 的瓶颈在 DRAM（93.31 % vs SM 4.98 %），闲置资源吃不掉这笔运费，所以不变慢。
但 `membound_qa.md` 的警告依然成立：**SM 利用率一上去，这笔运费就开始收税。**

---

## 5. "我需要知道每条指令的汇编实现吗"

**不需要背，需要会查、会验。**

| 层 | 内容 | 能穷举吗 | 掌握到什么程度 |
|---|---|---|---|
| PTX | 官方 ISA 手册的指令全集 | 能（唯一读得完的） | 认得常用的一二十条，其余查手册 |
| SASS | 芯片真正执行的机器码，每代不同 | **不能** | 会读访存/运算/分支几类，**怀疑时再看** |
| 硬件行为 | sector、事务、延迟、占用率 | 不是指令 | **最该花时间的一层** |

什么时候**必须**去看 SASS：

1. 指标异常：`sectors/请求 > 每线程字节数`（未对齐、跨步、发散）；
2. 怀疑编译器没向量化（该出 `.128` 却出 3 条 32 位）；
3. 寄存器用量或 spill 反常；
4. 想知道边界判断 / 分支的真实开销（`@P0 EXIT`、`BRA`）。

优先级永远是：**访问模式 > 占用率 > 指令混合 > 指令微调**。
`float3` 就是活证据：访存指令 3 倍，性能 0 变化。

> **SASS 是验尸报告，不是教材。**

---

## 6. 资料地图

### 官方（一手，按优先级）

| 资料 | 解决什么问题 | 地址 |
|---|---|---|
| CUDA C++ Programming Guide | 编程模型、内存层级、合并访存 | https://docs.nvidia.com/cuda/cuda-programming-guide/ |
| CUDA C++ Best Practices Guide | **优化优先级清单** | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ |
| PTX ISA | 指令全集 + 每条语义 | https://docs.nvidia.com/cuda/parallel-thread-execution/ |
| Turing Tuning Guide | 本机 sm_75 专属建议 | https://docs.nvidia.com/cuda/turing-tuning-guide/ |
| Nsight Compute Profiling Guide | **指标定义**（sector 就出自这里） | https://docs.nvidia.com/nsight-compute/ProfilingGuide/ |
| CUDA Binary Utilities | `cuobjdump` / `nvdisasm` 用法 | https://docs.nvidia.com/cuda/cuda-binary-utilities/ |

> 旧链接 `cuda-c-programming-guide` 现在 301 跳到 `cuda-programming-guide`，收藏别存错。

### 本机离线（不联网就能查，最实用）

```powershell
# 1. 指标全集 + 每个指标的一句话定义
ncu --query-metrics | Select-String "sectors_per_request"

# 2. ncu 报告里每个数字到底绑定了哪个指标 —— 理解报告的最短路径
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.1\sections\*.section"

# 3. CUDA v12.6\bin 里的全套离线工具
nvcc / ptxas / cuobjdump / nvdisasm
```

本笔记第 3 节就是这么查出来的：在 `MemoryWorkloadAnalysis.section` 里看到
`Label: "L2 Hit Rate" / Name: "lts__t_sector_hit_rate.pct"`。

### 二手

《Programming Massively Parallel Processors》（PMPP）当教材、
`NVIDIA/cuda-samples` 当题库、GPU MODE / CUDA MODE 社区当讨论区。
但都替代不了"用 ncu 测你自己的 kernel"。

---

## 7. 顺手修正前面笔记的两处说法

1. `03_factory_analogy.md` 说"float3 的指令数是 float4 的三倍"，
   精确说法是：**访存指令** 9 vs 3（3 倍），**总指令数** 24 vs 24（一样）。
   多出来的 6 条是加载/存储，不是全部指令。
2. `03` 说 `-cubin` 只编设备端代码、不需要 MSVC / vcvars。
   实测在干净 shell 里会报 `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`，
   `-cubin` 仍要走 host 预处理。**要么先 vcvars，要么用 `build.ps1`。**

---

## 8. 名词对照

| 英文 | 中文 | 白话 |
|---|---|---|
| sector | — | 32 B，最小搬运/记账单位 |
| cache line | 缓存行 | 128 B = 4 sectors |
| memory transaction | 内存事务 | 一次搬运，至少 1 个 sector |
| sectors per request | 每请求 sector 数 | 一条 warp 指令拆成几个 sector；理想 = 每线程字节数 |
| coalescing | 合并访存 | 让 32 个线程要的数据落在尽量少的 sector 里 |
| L1TEX | — | SM 里负责访存的单元；ncu 的 `l1tex__*` 指标就是它 |
| PTX / SASS | — | 跨代中间汇编 / 具体架构的机器码 |
| LDG / STG | — | SASS 的全局加载 / 存储，`.64` `.128` 是位宽 |

---

## 一句话收束

> **sector 是钱，指令是下单方式。先看账本（sector），再管怎么下单（指令）。**
