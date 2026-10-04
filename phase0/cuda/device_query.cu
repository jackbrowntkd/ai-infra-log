// 阶段 0 · 硬件参数查询
//
// 把 cudaDeviceProp 里跟"程序性能"相关的字段全部打出来。
// 这是整个项目后续所有性能讨论的硬件依据 —— 参数从这台机器实测，
// 不是抄 spec sheet。
//
// 注意：CUDA 的 runtime API 只暴露"每个 SM"级别的信息，
// 不会告诉你每个 SM 里有多少个 FP32 核心、多少个 LD/ST 单元。
// 那些要靠实测反推（见 phase0/calibrate.py 的 fp32 峰值）。

#include <cstdio>
#include <cuda_runtime.h>

static void line() { printf("--------------------------------------------------------------\n"); }

int main() {
    int dev = 0;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);

    printf("name                     %s\n", p.name);
    printf("compute capability       %d.%d\n", p.major, p.minor);
    printf("total global mem         %.2f GiB\n", p.totalGlobalMem / 1073741824.0);
    line();

    printf("SM count (SM 数量)       %d\n", p.multiProcessorCount);
    printf("warp size (warp 宽度)    %d threads\n", p.warpSize);
    printf("threads per SM           %d\n", p.maxThreadsPerMultiProcessor);
    printf("warps per SM (=T/SM/32)  %d\n", p.maxThreadsPerMultiProcessor / p.warpSize);
    printf("threads per block (max)  %d\n", p.maxThreadsPerBlock);
    printf("registers per SM         %d\n", p.regsPerMultiprocessor);
    printf("registers per thread mx  %d\n", p.regsPerBlock / p.maxThreadsPerBlock);
    printf("shared mem per SM        %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("shared mem per block     %zu KB (opt-in max %zu KB)\n",
           p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
    line();

    printf("L2 cache                 %d KB\n", p.l2CacheSize / 1024);
    printf("memory bus width         %d bits\n", p.memoryBusWidth);
    printf("memory clock             %.0f MHz\n", p.memoryClockRate / 1000.0);
    printf("SM clock (max)           %.0f MHz\n", p.clockRate / 1000.0);
    printf("concurrent kernels       %d\n", p.concurrentKernels);
    printf("async engines            %d\n", p.asyncEngineCount);
    line();

    // 以下是从上面推出来的，不是 API 直接给的
    const int fp32_per_sm = 64;   // Turing sm_75：每 SM 64 个 FP32 核心
    const double cores = (double)p.multiProcessorCount * fp32_per_sm;
    const double clk = p.clockRate * 1000.0;              // Hz
    const double bw = p.memoryBusWidth / 8.0 *
                      (p.memoryClockRate * 1000.0) * 2.0; // DDR：x2

    printf("DERIVED (推出来的，不是 API 给的)\n");
    printf("  FP32 cores             %.0f   (= %d SM x %d)\n", cores,
           p.multiProcessorCount, fp32_per_sm);
    printf("  FP32 peak @max clock   %.2f TFLOP/s  (= cores x 2 x clock)\n",
           cores * 2.0 * clk / 1e12);
    printf("  memory peak            %.1f GB/s  (= bus/8 x memclk x 2)\n",
           bw / 1e9);
    printf("  machine balance        %.1f FLOP/byte\n",
           cores * 2.0 * clk / bw);
    line();

    printf("\nper-SM resource budget (this is what decides occupancy)\n");
    printf("  threads   %5d  -> %d warps\n", p.maxThreadsPerMultiProcessor,
           p.maxThreadsPerMultiProcessor / p.warpSize);
    printf("  registers %5d  -> with R regs/thread, at most %d/R warps\n",
           p.regsPerMultiprocessor, p.regsPerMultiprocessor / p.warpSize);
    printf("  shared    %5zu KB -> with S KB/block, at most [SM shared]/S blocks\n",
           p.sharedMemPerMultiprocessor / 1024);
    return 0;
}