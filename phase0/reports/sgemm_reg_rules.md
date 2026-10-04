# ncu 官方规则报告 · sgemm_reg

- 采集时间：2026-10-04
- 被测程序：`phase0/cuda/sgemm.exe`
- 复现命令：`C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.1\ncu.BAT --set full -k regex:sgemm_reg --launch-skip 3 --launch-count 1 phase0/cuda/sgemm.exe`
- 规则总数：10（其中 5 条带收益估计）

> 这是 NVIDIA 规则引擎的判断，属于**症状**层面。因果链解释与手段选择见 `notes/09` 的杠杆表。

## 按预计收益排序

| # | 类型 | 预计收益 | 官方结论 | 出处 |
|---|---|---|---|---|
| 1 | OPT | **74.28 %** | The memory access pattern for global loads from L1TEX might not be optimal. On average, only 7.1 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global loads. | Memory Workload Analysis Tables |
| 2 | OPT | **71.63 %** | The memory access pattern for global stores to L1TEX might not be optimal. On average, only 8.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global stores. | Memory Workload Analysis Tables |
| 3 | OPT | **68.69 %** | This kernel has uncoalesced global accesses resulting in a total of 15073280 excessive sectors (78% of the total 19398656 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations. The CUDA Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) has additional information on reducing uncoalesced device memory accesses. | Source Counters |
| 4 | OPT | **52.25 %** | Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only issues an instruction every 5.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 8 warps per scheduler, this kernel allocates an average of 7.54 active warps per scheduler, but only an average of 0.38 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too. | Scheduler Statistics |
| 5 | OPT | **48.72 %** | On average, each warp of this kernel spends 19.3 cycles being stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure. This stall type represents about 48.7% of the total average of 39.7 cycles between issuing two instructions. | Warp State Statistics |

## 无收益估计的其它规则

- **OPT** [GPU Speed Of Light Throughput] This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.
- **INF** [GPU Speed Of Light Roofline Chart] The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The kernel achieved 26% of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline analysis.
- **INF** [Compute Workload Analysis] FMA is the highest-utilized pipeline (28.4%) based on active cycles, taking into account the rates of its different instructions. It executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations. It is well-utilized, but should not be a bottleneck.
- **WRN** [Memory Workload Analysis Chart] The optional metric lts__average_gcomp_input_sector_success_rate.pct could not be found. Collecting it as an additional metric could enable the rule to provide more guidance.
- **INF** [Warp State Statistics] Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on sampling data. The Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details on each stall reason.

## 分 section 明细

### GPU Speed Of Light Throughput

- **OPT** This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

### GPU Speed Of Light Roofline Chart

- **INF** The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The kernel achieved 26% of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline analysis.

### Compute Workload Analysis

- **INF** FMA is the highest-utilized pipeline (28.4%) based on active cycles, taking into account the rates of its different instructions. It executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations. It is well-utilized, but should not be a bottleneck.

### Memory Workload Analysis Chart

- **WRN** The optional metric lts__average_gcomp_input_sector_success_rate.pct could not be found. Collecting it as an additional metric could enable the rule to provide more guidance.

### Memory Workload Analysis Tables

- **OPT（Est. Speedup 74.28 %）** The memory access pattern for global loads from L1TEX might not be optimal. On average, only 7.1 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global loads.
- **OPT（Est. Speedup 71.63 %）** The memory access pattern for global stores to L1TEX might not be optimal. On average, only 8.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global stores.

### Scheduler Statistics

- **OPT（Est. Speedup 52.25 %）** Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only issues an instruction every 5.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 8 warps per scheduler, this kernel allocates an average of 7.54 active warps per scheduler, but only an average of 0.38 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.

### Warp State Statistics

- **OPT（Est. Speedup 48.72 %）** On average, each warp of this kernel spends 19.3 cycles being stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure. This stall type represents about 48.7% of the total average of 39.7 cycles between issuing two instructions.
- **INF** Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on sampling data. The Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details on each stall reason.

### Source Counters

- **OPT（Est. Speedup 68.69 %）** This kernel has uncoalesced global accesses resulting in a total of 15073280 excessive sectors (78% of the total 19398656 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations. The CUDA Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) has additional information on reducing uncoalesced device memory accesses.

