// 阶段 0 · 对照组：把 volatile 去掉，看编译器会不会把内存访问吃掉
//
// 问题：cache_probe 里的 add_reread 每轮迭代都真读内存 ——
//       这是硬件行为，还是被 volatile 逼出来的？
//
// 两个 kernel 逐字相同，唯一差别是 a / b 上有没有 volatile：
//   add_reread_vol    每轮必须真发一条 LDG（不许把读提到循环外）
//   add_reread_plain  a[i] + b[i] 不依赖 k，是循环不变量；循环体内没有任何
//                     写操作，所以把读提到循环外是合法且必然发生的优化，
//                     循环里只剩下寄存器 FADD
//
// 跑之前先写下预期，跑完逐条对账（见 reports/reg_reuse_qa.md）：
//   1. plain 在大 iters 档耗时塌下来（循环里不碰内存了）
//   2. plain 的 L1 / L2 sector 数掉到约 1 / iters
//   3. DRAM 字节数两边一样 —— vol 版的重读本来就命中 L2，DRAM 只付义务流量。
//      这条如果对上，等于再确认一次 cache_probe 的结论（DRAM 只认义务流量）。
//
// 用法：reg_reuse.exe            跑全部
//       reg_reuse.exe 0          只跑第 0 档（给 ncu 单档采数用）

#include "common.cuh"

// 对照组 A：volatile 版，和 cache_probe.cu 的 add_reread 逐字相同
__global__ void add_reread_vol(const volatile float* a, const volatile float* b,
                               float* __restrict__ c, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0.f;
#pragma unroll 1
        for (int k = 0; k < iters; ++k) {
            acc += a[i] + b[i];
        }
        c[i] = acc;
    }
}

// 对照组 B：唯一的变化就是去掉 volatile（函数签名其余部分、循环体、
// unroll 1 全部保持不变），这样"改了一个变量"这句话才成立
__global__ void add_reread_plain(const float* a, const float* b,
                                 float* __restrict__ c, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0.f;
#pragma unroll 1
        for (int k = 0; k < iters; ++k) {
            acc += a[i] + b[i];
        }
        c[i] = acc;
    }
}

// 用法：reg_reuse.exe            跑全部
//       reg_reuse.exe 3          只跑第 3 档（给 ncu 单档采数用）
int main(int argc, char** argv) {
    const int block = 256;
    int only = -1;                 // -1 = 全部
    if (argc > 1) only = atoi(argv[1]);

    struct Case { const char* tag; int n; int iters; };
    const Case cases[] = {
        {"64KB",   16 * 1024,        2048},   // 工作集 192 KB， 装得进 1 MB L2
        {"256KB",  64 * 1024,         512},   // 工作集 768 KB， 临界
        {"1MB",   256 * 1024,         128},   // 工作集 3 MB，  超出 L2
        {"64MB",  16 * 1024 * 1024,     2},   // 工作集 192 MB， 远超 L2
    };
    const int ncase = 4;

    print_header("volatile vs plain: does the compiler eat the memory traffic?");
    printf("  %-6s %8s %10s %10s %9s %12s %12s\n",
           "case", "iters", "vol ms", "plain ms", "speedup", "vol GB/s", "plain GB/s");

    for (int ci = 0; ci < ncase; ++ci) {
        if (only >= 0 && ci != only) continue;
        const int n = cases[ci].n;
        const int iters = cases[ci].iters;
        const size_t bytes = (size_t)n * sizeof(float);

        float *a, *b, *c;
        CUDA_CHECK(cudaMalloc(&a, bytes));
        CUDA_CHECK(cudaMalloc(&b, bytes));
        CUDA_CHECK(cudaMemset(a, 0, bytes));
        CUDA_CHECK(cudaMemset(b, 0, bytes));
        CUDA_CHECK(cudaMalloc(&c, bytes));

        const int grid = (n + block - 1) / block;
        // 逻辑读流量 = 代码"打算"读的字节数（不含写 c），两版共用同一个分母
        const double logical = 2.0 * (double)n * iters * sizeof(float);

        float ms_vol = bench_ms([&] {
            add_reread_vol<<<grid, block>>>(a, b, c, n, iters);
        }, 5, 20);

        float ms_plain = bench_ms([&] {
            add_reread_plain<<<grid, block>>>(a, b, c, n, iters);
        }, 5, 20);

        printf("  %-6s %8d %10.3f %10.3f %8.2fx %12.1f %12.1f\n",
               cases[ci].tag, iters, ms_vol, ms_plain, ms_vol / ms_plain,
               logical / (ms_vol * 1e-3) / 1e9,
               logical / (ms_plain * 1e-3) / 1e9);

        cudaFree(a); cudaFree(b); cudaFree(c);
    }

    // 义务流量：每个元素至少读 a、读 b、写 c 各一次，与 iters 无关
    print_header("obligatory DRAM traffic (one pass each of a, b, c)");
    for (int ci = 0; ci < ncase; ++ci) {
        if (only >= 0 && ci != only) continue;
        double oblig = 3.0 * (double)cases[ci].n * sizeof(float);
        printf("  %-6s  %10.2f MB\n", cases[ci].tag, oblig / 1e6);
    }
    return 0;
}
