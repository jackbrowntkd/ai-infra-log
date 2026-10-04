# ncu 官方规则报告 · sgemm_reg2

- 采集时间：2026-10-04
- 被测程序：`phase0/cuda/sgemm.exe`
- 复现命令：`C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.1\ncu.BAT --set full -k regex:sgemm_reg2 --launch-skip 3 --launch-count 1 phase0/cuda/sgemm.exe`
- 规则总数：10（其中 5 条带收益估计）

> 这是 NVIDIA 规则引擎的判断，属于**症状**层面。因果链解释与手段选择见 `notes/09` 的杠杆表。

## 按预计收益排序

| # | 类型 | 预计收益 | 官方结论 | 出处 |
|---|---|---|---|---|
| 1 | OPT | **47.11 %** | Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only issues an instruction every 2.7 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 8 warps per scheduler, this kernel allocates an average of 5.76 active warps per scheduler, but only an average of 0.73 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too. | Scheduler Statistics |
| 2 | OPT | **44.57 %** | On average, each warp of this kernel spends 7.1 cycles being stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure. This stall type represents about 44.6% of the total average of 15.8 cycles between issuing two instructions. | Warp State Statistics |
| 3 | OPT | **31.82 %** | The memory access pattern for shared stores might not be optimal and causes on average a 1.5 - way bank conflict across all 1048576 shared store requests.This results in 524288 bank conflicts,  which represent 33.33% of the overall 1572864 wavefronts for shared stores. Check the Source Counters section for uncoalesced shared stores. | Memory Workload Analysis Tables |
| 4 | OPT | **25.00 %** | The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 8. This kernel's theoretical occupancy (75.0%) is limited by the number of required registers. This kernel's theoretical occupancy (75.0%) is limited by the required amount of shared memory. | Occupancy |
| 5 | OPT | **3.60 %** | This kernel has uncoalesced shared accesses resulting in a total of 524288 excessive wavefronts (4% of the total 14155776 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations. The CUDA Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c -ab) has an example on optimizing shared memory accesses. | Source Counters |

## 无收益估计的其它规则

- **OPT** [GPU Speed Of Light Throughput] This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.
- **INF** [GPU Speed Of Light Roofline Chart] The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The kernel achieved 50% of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline analysis.
- **INF** [Compute Workload Analysis] FMA is the highest-utilized pipeline (54.4%) based on active cycles, taking into account the rates of its different instructions. It executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations. It is well-utilized, but should not be a bottleneck.
- **WRN** [Memory Workload Analysis Chart] The optional metric lts__average_gcomp_input_sector_success_rate.pct could not be found. Collecting it as an additional metric could enable the rule to provide more guidance.
- **INF** [Warp State Statistics] Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on sampling data. The Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details on each stall reason.

## 分 section 明细

### GPU Speed Of Light Throughput

- **OPT** This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

### GPU Speed Of Light Roofline Chart

- **INF** The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The kernel achieved 50% of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline analysis.

### Compute Workload Analysis

- **INF** FMA is the highest-utilized pipeline (54.4%) based on active cycles, taking into account the rates of its different instructions. It executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations. It is well-utilized, but should not be a bottleneck.

### Memory Workload Analysis Chart

- **WRN** The optional metric lts__average_gcomp_input_sector_success_rate.pct could not be found. Collecting it as an additional metric could enable the rule to provide more guidance.

### Memory Workload Analysis Tables

- **OPT（Est. Speedup 31.82 %）** The memory access pattern for shared stores might not be optimal and causes on average a 1.5 - way bank conflict across all 1048576 shared store requests.This results in 524288 bank conflicts,  which represent 33.33% of the overall 1572864 wavefronts for shared stores. Check the Source Counters section for uncoalesced shared stores.

### Scheduler Statistics

- **OPT（Est. Speedup 47.11 %）** Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only issues an instruction every 2.7 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 8 warps per scheduler, this kernel allocates an average of 5.76 active warps per scheduler, but only an average of 0.73 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.

### Warp State Statistics

- **OPT（Est. Speedup 44.57 %）** On average, each warp of this kernel spends 7.1 cycles being stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure. This stall type represents about 44.6% of the total average of 15.8 cycles between issuing two instructions.
- **INF** Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on sampling data. The Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details on each stall reason.

### Occupancy

- **OPT（Est. Speedup 25.00 %）** The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 8. This kernel's theoretical occupancy (75.0%) is limited by the number of required registers. This kernel's theoretical occupancy (75.0%) is limited by the required amount of shared memory.

### Source Counters

- **OPT（Est. Speedup 3.60 %）** This kernel has uncoalesced shared accesses resulting in a total of 524288 excessive wavefronts (4% of the total 14155776 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations. The CUDA Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c -ab) has an example on optimizing shared memory accesses.

