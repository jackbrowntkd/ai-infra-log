// 阶段 0 · 延迟隐藏：并行度、在途请求数与 ILP
//
// 回答两个问题：
//   EXP1  并行度低到什么程度，DRAM 会从"打满"掉到"空转"？
//   EXP2  用 ILP（每线程多笔独立请求）能不能把并行度不足补回来？
//
// 设计：纯读一个 240 MB 数组（float4），无复用、无写。
//   read_gs     每线程一次只挂 1 笔请求
//   read_gs_ilp4 每线程一次挂 4 笔独立请求（4 倍在途量）

#include "common.cuh"

__global__ void read_gs(const float4* __restrict__ a, float* __restrict__ out, int n4) {
    int stride = gridDim.x * blockDim.x;
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = a[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;   // 防止整个循环被优化掉
}

__global__ void read_gs_ilp4(const float4* __restrict__ a, float* __restrict__ out, int n4) {
    int stride = gridDim.x * blockDim.x;
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    int i = start;
    for (; i + 3 * stride < n4; i += 4 * stride) {
        float4 v0 = a[i];
        float4 v1 = a[i + stride];
        float4 v2 = a[i + 2 * stride];
        float4 v3 = a[i + 3 * stride];
        acc += v0.x + v0.y + v0.z + v0.w;
        acc += v1.x + v1.y + v1.z + v1.w;
        acc += v2.x + v2.y + v2.z + v2.w;
        acc += v3.x + v3.y + v3.z + v3.w;
    }
    for (; i < n4; i += stride) {
        float4 v = a[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;
}

int main() {
    const int block = 256;
    const int n = 60 * 1024 * 1024;            // 60M floats = 240 MB
    const int n4 = n / 4;
    const size_t bytes = (size_t)n * sizeof(float);

    float4* a;
    float* out;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMemset(a, 0, bytes));
    CUDA_CHECK(cudaMalloc(&out, sizeof(float)));

    const double moved = (double)bytes;        // 纯读 240 MB
    const int grids[] = {1, 2, 4, 7, 14, 28, 56, 112};

    print_header("EXP1  grid sweep, 每线程 1 笔在途请求 (240 MB 纯读)");
    printf("  %-8s %8s %14s %12s %10s\n",
           "grid", "warps", "在途(估算)", "ms", "GB/s");
    for (int gi = 0; gi < 8; ++gi) {
        const int grid = grids[gi];
        const int warps = grid * block / 32;
        const double inflight = (double)warps * 512.0;      // 每 warp 1 笔 x 512 B
        float ms = bench_ms([&] { read_gs<<<grid, block>>>(a, out, n4); }, 3, 10);
        printf("  %-8d %8d %11.0f KB %12.3f %10.1f\n",
               grid, warps, inflight / 1024.0, ms, moved / (ms * 1e-3) / 1e9);
    }

    print_header("EXP2  同上，但每线程 4 笔独立请求 (ILP=4)");
    printf("  %-8s %8s %14s %12s %10s\n",
           "grid", "warps", "在途(估算)", "ms", "GB/s");
    for (int gi = 0; gi < 8; ++gi) {
        const int grid = grids[gi];
        const int warps = grid * block / 32;
        const double inflight = (double)warps * 4 * 512.0;
        float ms = bench_ms([&] { read_gs_ilp4<<<grid, block>>>(a, out, n4); }, 3, 10);
        printf("  %-8d %8d %11.0f KB %12.3f %10.1f\n",
               grid, warps, inflight / 1024.0, ms, moved / (ms * 1e-3) / 1e9);
    }

    cudaFree(a);
    cudaFree(out);
    return 0;
}
