// 阶段 0 · 数据量 / 复用距离 对缓存和带宽的影响
//
// 回答三个问题：
//   EXP1  逻辑工作量固定，改变每个数组的大小时，有效带宽怎么变？
//   EXP2  同样多的活，"1 次启动 + 内部循环" vs "K 次启动" 差多少？
//   EXP3  给 roofline 用的点：迭代次数增加时，AI 怎么走
//
// 关键设计：每次迭代都从全局内存重新读（volatile + unroll 1），
// 编译器无法把读提到循环外，所以"是否命中缓存"完全由数据量决定。

#include "common.cuh"
#include <cstdlib>
#include <cstring>

__global__ void add_reread(const volatile float* a, const volatile float* b,
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

// 每个线程处理 k 个元素（线程内跨 stride，warp 内保持连续 -> 仍然合并）。
// k 越大，"两次读到同一个地址之间"隔的数据越多 = 复用距离越长。
__global__ void add_reread_k(const volatile float* a, const volatile float* b,
                             float* __restrict__ c, int n, int k, int stride,
                             int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
#pragma unroll 1
    for (int t = 0; t < iters; ++t) {
#pragma unroll 1
        for (int j = 0; j < k; ++j) {
            int idx = tid + j * stride;
            if (idx < n) acc += a[idx] + b[idx];
        }
    }
    if (tid < n) c[tid] = acc;
}

// 用法：cache_probe.exe            跑全部
//       cache_probe.exe 2         只跑第 2 档（给 ncu 单档采数用）
//       cache_probe.exe bs        跑 block size 扫描（1MB 档）
//       cache_probe.exe bs 768    只跑 block=768（给 ncu 单点采数用）
//       cache_probe.exe k         跑 k 扫描（调复用距离跨过 1MB L2）
//       cache_probe.exe k 12 768  只跑 k=12, block=768（给 ncu 用）
int main(int argc, char** argv) {
    const int block = 256;
    int only = -1;                 // -1 = 全部
    const char* a1 = (argc > 1) ? argv[1] : "";
    const bool bs_mode = (strcmp(a1, "bs") == 0);
    const bool k_mode = (strcmp(a1, "k") == 0);
    int only_block = 0;            // 0 = 全部
    int only_k = 0;
    if (bs_mode) {
        if (argc > 2) only_block = atoi(argv[2]);
    } else if (k_mode) {
        if (argc > 2) only_k = atoi(argv[2]);
        if (argc > 3) only_block = atoi(argv[3]);
    } else if (argc > 1) {
        only = atoi(a1);
    }

    // EXP4：调复用距离，跨过 1 MB 的 L2 容量
    if (k_mode) {
        const int sm_threads = 14 * 1024;      // 一整波 = 14 SM x 1024 线程
        const int iters = 64;
        const int ks[] = {2, 4, 8, 12, 16};
        const int blks[] = {256, 768};         // 100 % vs 74.6 % 占用率
        const int nmax = sm_threads * 16 + 16384;
        const size_t bytes = (size_t)nmax * sizeof(float);

        float *a, *b, *c;
        CUDA_CHECK(cudaMalloc(&a, bytes));
        CUDA_CHECK(cudaMalloc(&b, bytes));
        CUDA_CHECK(cudaMemset(a, 0, bytes));
        CUDA_CHECK(cudaMemset(b, 0, bytes));
        CUDA_CHECK(cudaMalloc(&c, bytes));

        print_header("EXP4  reuse distance vs L2 capacity (1 MB)");
        printf("  %-7s %-7s %8s %14s %12s %10s\n",
               "block", "k", "ms", "reuse-dist", "GB/s(log)", "resident");
        for (int bi = 0; bi < 2; ++bi) {
            const int blk = blks[bi];
            if (only_block && blk != only_block) continue;
            const int grid = (sm_threads + blk - 1) / blk;
            const int stride = grid * blk;
            // 常驻线程数：能整除时满占用；768 只能住 1 个 block/SM
            const int per_sm = (1024 / blk) > 32 ? 32 : (1024 / blk);
            const int resident = 14 * per_sm * blk;
            for (int ki = 0; ki < 5; ++ki) {
                const int k = ks[ki];
                if (only_k && k != only_k) continue;
                const int n = stride * k;
                float ms = bench_ms(
                    [&] { add_reread_k<<<grid, blk>>>(a, b, c, n, k, stride, iters); },
                    5, 20);
                double gbps = 2.0 * n * iters * 4 / (ms * 1e-3) / 1e9;
                double rd = (double)resident * k * 8.0;
                printf("  %-7d %-7d %8.3f %11.0f KB %12.1f %10d\n",
                       blk, k, ms, rd / 1024.0, gbps, resident);
            }
        }
        cudaFree(a); cudaFree(b); cudaFree(c);
        return 0;
    }

    // EXP3：只改 block size，看占用率 -> 复用距离 -> 命中率 这条链
    if (bs_mode) {
        const int n = 256 * 1024;      // 1 MB / 数组，工作集 3 MB > 1 MB L2
        const int iters = 128;
        const size_t bytes = (size_t)n * sizeof(float);
        float *a, *b, *c;
        CUDA_CHECK(cudaMalloc(&a, bytes));
        CUDA_CHECK(cudaMalloc(&b, bytes));
        CUDA_CHECK(cudaMemset(a, 0, bytes));
        CUDA_CHECK(cudaMemset(b, 0, bytes));
        CUDA_CHECK(cudaMalloc(&c, bytes));

        print_header("EXP3  block size sweep (1MB arrays, iters=128)");
        printf("  %-8s %10s %10s %14s\n", "block", "grid", "ms", "GB/s(logical)");
        const int blocks[] = {256, 512, 1024, 192, 768};
        for (int bi = 0; bi < 5; ++bi) {
            const int blk = blocks[bi];
            if (only_block && blk != only_block) continue;
            const int grid = (n + blk - 1) / blk;
            float ms = bench_ms([&] { add_reread<<<grid, blk>>>(a, b, c, n, iters); }, 5, 20);
            double gbps = 2.0 * n * iters * 4 / (ms * 1e-3) / 1e9;
            printf("  %-8d %10d %10.3f %14.1f\n", blk, grid, ms, gbps);
        }
        cudaFree(a); cudaFree(b); cudaFree(c);
        return 0;
    }

    // 每个数组 n 个 float；n * iters 固定为 33.5M 次重读
    struct Case { const char* tag; int n; int iters; };
    const Case cases[] = {
        {"64KB",   16 * 1024,        2048},
        {"256KB",  64 * 1024,         512},
        {"1MB",   256 * 1024,         128},
        {"64MB",  16 * 1024 * 1024,     2},
    };

    Point pts[8];
    int np = 0;

    print_header("EXP1  data size x fixed logical work (33.5M re-reads)");
    printf("  %-8s %10s %8s %10s %14s %12s\n",
           "size", "floats", "iters", "ms", "GB/s(logical)", "DRAM fit?");

    for (int ci = 0; ci < 4; ++ci) {
        if (only >= 0 && ci != only) continue;
        const int n = cases[ci].n;
        const int iters = cases[ci].iters;
        const size_t bytes = (size_t)n * sizeof(float);
        const double logical = 2.0 * (double)n * iters * sizeof(float);

        float *a, *b, *c;
        CUDA_CHECK(cudaMalloc(&a, bytes));
        CUDA_CHECK(cudaMalloc(&b, bytes));
        CUDA_CHECK(cudaMemset(a, 0, bytes));
        CUDA_CHECK(cudaMemset(b, 0, bytes));
        CUDA_CHECK(cudaMalloc(&c, bytes));

        const int grid = (n + block - 1) / block;
        float ms = bench_ms([&] { add_reread<<<grid, block>>>(a, b, c, n, iters); },
                            5, 20);
        double gbps = logical / (ms * 1e-3) / 1e9;
        double flops = 2.0 * (double)n * iters;
        double gflops = flops / (ms * 1e-3) / 1e9;

        printf("  %-8s %10d %8d %10.3f %14.1f %12s\n", cases[ci].tag, n, iters, ms,
               gbps, (3.0 * bytes < 1e6) ? "yes(192KB)" : "no");

        // gbps 先留 0，等 ncu 量出真实 DRAM 字节数再回填
        pts[np].name = cases[ci].tag;
        pts[np].ms = ms;
        pts[np].gflops = gflops;
        pts[np].gbps = 0.0;
        pts[np].ai = 0.0;
        ++np;

        cudaFree(a); cudaFree(b); cudaFree(c);
    }

    // EXP2：同样是 2048 轮重读，一次启动 vs 2048 次启动
    print_header("EXP2  one launch with inner loop  vs  K launches");
    if (only < 0) {
        const int n = 16 * 1024;
        const int iters = 2048;
        const size_t bytes = (size_t)n * sizeof(float);
        float *a, *b, *c;
        CUDA_CHECK(cudaMalloc(&a, bytes));
        CUDA_CHECK(cudaMalloc(&b, bytes));
        CUDA_CHECK(cudaMemset(a, 0, bytes));
        CUDA_CHECK(cudaMemset(b, 0, bytes));
        CUDA_CHECK(cudaMalloc(&c, bytes));
        const int grid = (n + block - 1) / block;

        float ms_one = bench_ms([&] {
            add_reread<<<grid, block>>>(a, b, c, n, iters);
        }, 5, 20);

        float ms_many = bench_ms([&] {
            for (int k = 0; k < iters; ++k) {
                add_reread<<<grid, block>>>(a, b, c, n, 1);
            }
        }, 1, 3);

        printf("  1 次启动 + 内部循环 2048 轮 : %10.3f ms\n", ms_one);
        printf("  2048 次启动, 每次 1 轮      : %10.3f ms\n", ms_many);
        printf("  每次启动的额外开销          : %10.3f us\n",
               (ms_many - ms_one) / iters * 1000.0);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }

    // 只在跑全部 4 档时写盘，避免单档采数把完整结果覆盖掉
    if (only < 0) {
        write_points("phase0/measured/cache_probe.json", pts, np);
    }
    return 0;
}
