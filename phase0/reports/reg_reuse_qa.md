# volatile 对照组：编译器会在你不设防时把访存吃掉 —— 实测报告

配套：`phase0/cuda/reg_reuse.cu`
原始输出：`phase0/reports/reg_reuse_sass.txt`（SASS）、`phase0/reports/reg_reuse_ncu.txt`（计数器）

**要回答的问题**：`cache_probe.cu` 里的 `add_reread` 每轮迭代都真读一次内存——
这是硬件行为，还是被 `volatile` 逼出来的？

**设计**：两个 kernel 逐字相同（同样的循环体、同样的 `#pragma unroll 1`），
唯一差别是 `a` / `b` 上有没有 `volatile`。四档数据量与 `cache_probe` 一致
（`n x iters` 恒定在 33.5M 次重读）。

---

## 一、结果总表

| 档 | iters | vol ms | plain ms | 加速比 | vol 逻辑 GB/s | plain 逻辑 GB/s |
|---|---|---|---|---|---|---|
| 64KB | 2048 | 1.168 | 0.101 | **11.60x** | 229.9 | **2665.6** |
| 256KB | 512 | 1.006 | 0.078 | **12.91x** | 266.9 | **3445.1** |
| 1MB | 128 | 0.929 | 0.081 | **11.49x** | 289.1 | **3321.1** |
| 64MB | 2 | 1.286 | 1.128 | 1.14x | 208.8 | 238.1 |

> "逻辑 GB/s" = 代码**打算**读的字节数 / 时间。本机实测带宽峰值是 179 GB/s
> （`hardware.json`）。plain 那三档算出 2600~3400 GB/s —— 见 §五 第 1 条。

---

## 二、SASS：读被提到了循环外（决定性证据）

**plain 版**（`add_reread_plain`）：

```text
/*0110*/  LDG.E.CONSTANT.SYS R0, [R4.64+UR4] ;   <- 循环外，全程只发一次
/*0120*/  LDG.E.CONSTANT.SYS R9, [R4.64+UR6] ;
/*0130*/  CS2R R6, SRZ ;                          <- 循环计数清零
/*0140*/  FADD R0, R0, R9 ;                       <- a+b 也被提到循环外
          loop:                                   <- 循环体：4 条指令，零访存
/*0150*/  IADD3 R6, R6, 0x1, RZ ;
/*0160*/  ISETP.GE.AND P0, PT, R6, c[0x0][0x17c], PT ;
/*0170*/  FADD R7, R0, R7 ;                       <- 只剩累加
/*0180*/ @!P0 BRA 0x150 ;
/*0190*/  STG.E.SYS [R2], R7 ;
```

**vol 版**（`add_reread_vol`，其余代码完全相同）：

```text
          loop:                                   <- 循环体：含两条 LDG
/*0110*/  ULDC.64 UR4, c[0x0][0x168] ;           <- 连"取地址"都在循环里
/*0130*/  LDG.E.STRONG.SYS R6, [R4.64+UR4] ;     <- 每轮真读
/*0140*/  LDG.E.STRONG.SYS R9, [R4.64+UR6] ;
/*0150*/  IADD3 R0, R0, 0x1, RZ ;
/*0160*/  ISETP.GE.AND P0, PT, R0, c[0x0][0x17c], PT ;
/*0170*/  FADD R6, R6, R9 ;
/*0180*/  FADD R7, R6, R7 ;
/*0190*/ @!P0 BRA 0x110 ;                         <- 回到 0x110
```

三个可核对的事实：

1. **vol 的循环体含两条 LDG；plain 的循环体一条访存都没有。** 两次 LDG 的地址
   `[R4.64+UR4]`、`[R4.64+UR6]` 在循环外算好、循环内不变 —— 就是"反复读同一格"。
2. **opcode 不同**：vol 是 `LDG.E.STRONG.SYS`（强内存序），plain 是
   `LDG.E.CONSTANT.SYS`（只读路径，即 `__ldg` 那条）。**同一行 CUDA 代码、
   两种硬件行为** —— 这也解释了笔记 08 §3 的对账表为什么分母必须是 5：
   `STRONG` 语义的读不会被 L1 悄悄接住。
3. **循环本身没被吃掉**：plain 仍是 2048 次 FADD + 循环开销。FP 不可重结合，
   编译器不能把 `acc += t` 2048 次折叠成一次乘法。这一条马上用耗时验证。

---

## 三、ncu：sector 数按 iters 倍缩水，DRAM 字节数纹丝不动

| 档 | kernel | gpu 时长 | DRAM 读 | DRAM 写 | L1 读 sector | L2 命中率 |
|---|---|---|---|---|---|---|
| 64KB | vol | 1.19 ms | 133.76 KB | 49.95 KB | **8,388,608** | 99.95 % |
| 64KB | plain | **100.16 us** | 133.76 KB | 9.86 KB | **4,096** | 41.09 % |
| 64MB | vol | 1.29 ms | 134.23 MB | 67.22 MB | **8,388,608** | 60.00 % |
| 64MB | plain | 1.13 ms | 134.22 MB | 67.08 MB | **4,194,304** | 33.35 % |

**三条结论：**

1. **L1 读 sector 数之比恰好等于 iters**：64KB 档 8,388,608 ÷ 4,096 = **2048**；
   64MB 档 8,388,608 ÷ 4,194,304 = **2**。这是"读被提到循环外"的定量指纹，
   和 §二 的 SASS 完全咬合。
2. **DRAM 字节数两边完全相同**（读 133.76 KB vs 133.76 KB；134.23 MB vs 134.22 MB）。
   → 再确认一次 `cache_probe` 的结论：vol 版的重读**全部命中 L2**，
   DRAM 只付义务流量。**去掉 volatile 省下的是 L2 的活，不是 DRAM 的活。**
3. **两档 vol 的 sector 数一模一样（8,388,608）**：因为 `n x iters` 恒定，
   逻辑工作量被钉死了 —— 实验设计本身在数据里显形。

顺带把 64MB 档的命中率对上账：vol = **60.00 %**（= 3/5，笔记 08 §3 那个对账表），
plain = **33.35 %**（= 1/3，全部来自"写满 sector 的 store 算命中"，读命中为零）。
**同一份代码、同一个数据，两个命中率把"有没有复用"这件事的两端都钉住了。**

> 64KB 档 plain 的 41.09 % 略高于 1/3：它连着跑了 25 次、工作集只有 192 KB，
> 上一次启动留在 L2 里的 a、b 被下一次碰到 —— 是**跨启动**的复用，
> 和 vol 那种**同一趟 kernel 内**的复用不是一回事。

### 校准：我最初猜错了一条

| 跑之前的预期 | 实测 |
|---|---|
| plain 在大 iters 档耗时塌下来 | ✅ 11.5 ~ 12.9 倍 |
| plain 的 L1 sector 数掉到约 1/iters | ✅ 精确 |
| DRAM 字节数两边一样 | ✅ |
| ~~DRAM 字节数掉到接近 0~~ | ❌ **错**，见 §三 第 2 条 |

错的原因是忘了 `cache_probe` 已经证明过的事：**vol 版的重读本来就命中 L2**，
DRAM 从不重复付账。所以"去掉 volatile 能让 DRAM 流量归零"这个想法，
在这个实验里从一开始就不成立。

---

## 四、耗时差在哪：11~13 倍 vs 1.14 倍

| 档 | 加速比 | 为什么 |
|---|---|---|
| 64KB / 256KB / 1MB | **11.5 ~ 12.9x** | 复用全被 L2 接住，vol 的时间几乎全花在"每轮再发两条 LDG"上；去掉就是净赚 |
| 64MB | **1.14x** | 这一档 DRAM 本来就接近饱和（81 %），两版搬一样的字节，删掉 L2 层的活省不下多少 |

**这个差距量化了"缓存替你干了多少活"。** 同一个 kernel、同样的逻辑工作量，
只因为工作集大小不同，缓存贡献的加速从 12 倍缩到 1.1 倍。

plain 版没有归零到 0.02 ms，而是停在 0.100 ms —— 因为循环还在（§二 第 3 条）。
**编译器只吃掉了访存，没吃掉 2048 次 FADD。**

---

## 五、两个副作用，都是有用的信号

### 1. 逻辑带宽出现了物理上不可能的数

plain 版 64KB 档"逻辑带宽" = **2665 GB/s**，是本机实测峰值 179 GB/s 的 **14.9 倍**。

> **判据：逻辑带宽一旦超过硬件峰值，只可能是"你以为在读，其实没读"。**
> 反过来也成立：任何"我读了 N 遍"的推算，先用 **SASS 或 sector 数**确认读真的发生了。

### 2. ncu 坑 1 的复现（笔记 08 §6）

profiler 下程序自报的 vol 耗时是 **17.7 ms**（独立运行 1.168 ms，**慢 15 倍**），
但 `gpu__time_duration.sum` 仍是 **1.19 ms**，和独立运行一致。

> **时间取独立运行。要看 kernel 本身的时长就用 `gpu__time_duration`，
> 别用程序自报的 elapsed —— 后者把 profiler 的回放和插桩开销一起算进去了。**

---

## 六、对写 kernel 的三条影响

1. **`volatile` 不是"让程序更正确"的银弹**，它同时关掉了编译器最强的一个优化。
   只在"必须真的重读"时才用：测量探针、跨设备通信、调试。
2. **寄存器复用是编译器免费送的。** 凡是它能证明"值不变"的地方，访存都会被省掉。
   缓存存在的意义，正是它**证不出来**的场景：跨线程（warp 内 lane 之间读不到
   对方的寄存器）、跨 block、跨阶段（寄存器压力活不到那时候）。
3. **`cache_probe` 的设计因此是必要的**：不加 `volatile`，那个 kernel 测的就是
   寄存器加法速度，和内存层级一点关系都没有。

---

## 七、复现命令

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File phase0\cuda\build.ps1 phase0\cuda\reg_reuse.cu
.\phase0\cuda\reg_reuse.exe          # 四档全跑
.\phase0\cuda\reg_reuse.exe 0        # 只跑第 0 档（给 ncu 单档采数用）

# SASS 对照（不需要管理员）
cuobjdump -sass phase0\cuda\reg_reuse.exe > phase0\reports\reg_reuse_sass.txt

# 计数器（需要管理员 shell）
ncu --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sector_hit_rate.pct `
    -k regex:add_reread_vol --launch-skip 5 --launch-count 1 .\phase0\cuda\reg_reuse.exe 0
```

> 换了机器要重跑：本机 L2 只有 1 MB，三档小工作集"装得进 L2"这个前提是结论的一部分。
