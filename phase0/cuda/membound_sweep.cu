// 阶段 0 · memory-bound 的三个扫描实验
//
// 回答三个问题：
//   1. 向量宽度 f1/f2/f3/f4 到底有没有区别？
//   2. block size 有没有区别？
//   3. 并行度要多少才够打满带宽？（grid 扫描）
//
// 字节数在所有变体里都是 3 * 240 MB = 720 MB，算术强度恒为 1/12，
// 所以任何性能差异都只能来自「访存指令的组织方式」，与计算无关。

#include "common.cuh"

__global__ void add_f1(const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void add_f2(const float2* __restrict__ a, const float2* __restrict__ b,
                       float2* __restrict__ c, int n2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) {
        float2 x = a[i], y = b[i];
        c[i] = make_float2(x.x + y.x, x.y + y.y);
    }
}

// float3 是 12 字节，对齐只有 4 字节。编译器没法生成一条 128 位访存指令，
// 只能拆成 3 条 32 位。预期它比 f1 还慢。
__global__ void add_f3(const float3* __restrict__ a, const float3* __restrict__ b,
                       float3* __restrict__ c, int n3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n3) {
        float3 x = a[i], y = b[i];
        c[i] = make_float3(x.x + y.x, x.y + y.y, x.z + y.z);
    }
}

__global__ void add_f4(const float4* __restrict__ a, const float4* __restrict__ b,
                       float4* __restrict__ c, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

// grid-stride 版本：固定 block 数，每个线程循环处理多个元素。
// 用来扫「需要多少并行度才能打满带宽」。
__global__ void add_f4_stride(const float4* __restrict__ a,
                              const float4* __restrict__ b,
                              float4* __restrict__ c, int n4) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

int main() {
    const int n = 62914560;                    // 60M floats，能被 12 整除
    const size_t bytes = (size_t)n * sizeof(float);
    const double moved = 3.0 * (double)bytes;  // 720 MB

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMemset(a, 0, bytes));
    CUDA_CHECK(cudaMemset(b, 0, bytes));

    const int n1 = n, n2 = n / 2, n3 = n / 3, n4 = n / 4;

    print_header("EXP1  vector width  (720 MB per launch, AI = 1/12)");
    printf("  %-8s %-8s %10s %12s %12s\n", "variant", "block", "ms", "GB/s", "GB/s per thread");
    {
        struct { const char* tag; int vec; } kinds[] = {
            {"f1", 1}, {"f2", 2}, {"f3", 3}, {"f4", 4}};
        for (auto k : kinds) {
            int block = 256;
            int cnt = n / k.vec;
            int grid = (cnt + block - 1) / block;
            float ms = bench_ms([&] {
                if (k.vec == 1) add_f1<<<grid, block>>>(a, b, c, n1);
                else if (k.vec == 2) add_f2<<<grid, block>>>((const float2*)a, (const float2*)b, (float2*)c, n2);
                else if (k.vec == 3) add_f3<<<grid, block>>>((const float3*)a, (const float3*)b, (float3*)c, n3);
                else add_f4<<<grid, block>>>((const float4*)a, (const float4*)b, (float4*)c, n4);
            });
            double gbps = moved / (ms * 1e-3) / 1e9;
            printf("  %-8s %-8d %10.3f %12.1f %12.2f\n", k.tag, block, ms, gbps,
                   12.0 / k.vec);
        }
    }

    print_header("EXP2  block size sweep  (float4)");
    printf("  %-8s %10s %12s\n", "block", "ms", "GB/s");
    for (int block : {64, 128, 256, 512, 1024}) {
        int grid = (n4 + block - 1) / block;
        float ms = bench_ms([&] {
            add_f4<<<grid, block>>>((const float4*)a, (const float4*)b, (float4*)c, n4);
        });
        printf("  %-8d %10.3f %12.1f\n", block, ms, moved / (ms * 1e-3) / 1e9);
    }

    print_header("EXP3  grid size sweep  (float4, block=256, grid-stride)");
    printf("  %-10s %10s %10s %12s\n", "grid", "warps", "ms", "GB/s");
    int block = 256;
    for (int grid : {14, 28, 56, 112, 224, 448, 896, 1792, 3584, 7168, 16384}) {
        float ms = bench_ms([&] {
            add_f4_stride<<<grid, block>>>((const float4*)a, (const float4*)b, (float4*)c, n4);
        });
        printf("  %-10d %10d %10.3f %12.1f\n", grid, grid * block / 32, ms,
               moved / (ms * 1e-3) / 1e9);
    }

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}