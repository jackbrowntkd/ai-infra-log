// 阶段 0 · 第一个 kernel：向量加
//
// 目标不是"写出来"，是"看懂它在 roofline 上的位置"。
//
// 向量加是极端的 memory-bound：搬 12 字节只做 1 次浮点加法，
// 算术强度 1/12 = 0.083 FLOP/byte，离本机 15.4 的转折点差两个数量级。
// 所以优化它的唯一方向是"把带宽吃满"，而不是"减少计算"。
//
// 这里对比两版，外加一个 block size 扫描：
//   naive   —— 每个线程处理 1 个 float，标量访存
//   float4  —— 每个线程处理 4 个 float，一次 128 位访存
//
// 注意：本文件的字符串字面量必须全用 ASCII。nvcc 处理不了字符串里的
// 非 ASCII 字符，会报 "missing closing quote"；注释里写中文则没问题，
// 但文件必须存成带 BOM 的 UTF-8，否则 MSVC 会按 GBK 解析。

#include "common.cuh"

__global__ void vecadd_naive(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vecadd_float4(const float4* __restrict__ a,
                              const float4* __restrict__ b,
                              float4* __restrict__ c, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i];
        float4 y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

// 初始化数据。不能用 cudaMemset —— 它是按字节填的，
// 填出来的 0x01010101 是个非规格化浮点数，不是 1.0。
__global__ void init_data(float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = (float)(i % 1000) * 0.25f;
        b[i] = (float)(i % 97);
    }
}

__device__ __host__ inline float expect_at(int i) {
    return (float)(i % 1000) * 0.25f + (float)(i % 97);
}

int main() {
    const int n = 64 << 20;                    // 64M floats = 每个数组 256 MB
    const size_t bytes = (size_t)n * sizeof(float);
    const double moved = 3.0 * (double)bytes;  // 读 a + 读 b + 写 c
    const int n4 = n / 4;

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));

    {
        int block = 256;
        int grid = (n + block - 1) / block;
        init_data<<<grid, block>>>(a, b, n);
        CUDA_CHECK(cudaGetLastError());
    }

    Point pts[8];
    int np = 0;

    print_header("vector add  N=64M floats, 768 MB per launch");

    {
        int block = 256;
        int grid = (n + block - 1) / block;
        float ms = bench_ms([&] { vecadd_naive<<<grid, block>>>(a, b, c, n); });
        double gbps = moved / (ms * 1e-3) / 1e9;
        printf("  %-10s block=%-5d %8.3f ms   %7.1f GB/s\n", "naive", block, ms, gbps);
        pts[np++] = {"vecadd_naive", ms, 0.0, gbps, 1.0 / 12.0};
    }

    for (int block : {128, 256, 512, 1024}) {
        int grid = (n4 + block - 1) / block;
        float ms = bench_ms([&] {
            vecadd_float4<<<grid, block>>>((const float4*)a, (const float4*)b,
                                           (float4*)c, n4);
        });
        double gbps = moved / (ms * 1e-3) / 1e9;
        printf("  %-10s block=%-5d %8.3f ms   %7.1f GB/s\n", "float4", block, ms, gbps);
        if (block == 256) {
            char* name = (char*)malloc(32);
            snprintf(name, 32, "vecadd_float4_b%d", block);
            pts[np++] = {name, ms, 0.0, gbps, 1.0 / 12.0};
        }
    }

    print_header("correctness");

    vecadd_float4<<<(n4 + 255) / 256, 256>>>((const float4*)a, (const float4*)b,
                                             (float4*)c, n4);
    CUDA_CHECK(cudaGetLastError());

    float* h = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h, c, bytes, cudaMemcpyDeviceToHost));

    int bad = 0;
    float worst = 0.f;
    for (int i = 0; i < n; i += 4096) {
        float diff = fabsf(h[i] - expect_at(i));
        if (diff > worst) worst = diff;
        if (diff > 1e-3f) ++bad;
    }
    printf("  checked %d samples, mismatches=%d, max abs err=%.3e -> %s\n",
           n / 4096, bad, worst, bad == 0 ? "PASS" : "FAIL");

    write_points("phase0/measured/vecadd.json", pts, np);

    free(h);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}