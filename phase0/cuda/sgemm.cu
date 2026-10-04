// 阶段 0 · 矩阵乘法：从 naive 到 shared memory tiling
//
// 对比两个版本（N=1024，A/B/C 各 4 MB，工作集 12 MB >> 1 MB L2）：
//   sgemm_naive  每个线程算一个 C 元素，k 循环直接从全局内存读
//   sgemm_tiled  32x32 分块，先把 A/B 的 tile 搬进共享内存再算
//
// 预期：naive 被 DRAM 卡死，tiled 把每个元素的全局读取次数降到 1/32。

#include "common.cuh"

#define TILE 32

__global__ void sgemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k) {
            acc += A[(size_t)row * N + k] * B[(size_t)k * N + col];
        }
        C[(size_t)row * N + col] = acc;
    }
}

__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < N / TILE; ++t) {
        // 搬运（全局 -> 共享）：每个线程搬 1 个元素，两趟
        As[threadIdx.y][threadIdx.x] = A[(size_t)row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[((size_t)(t * TILE + threadIdx.y)) * N + col];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[(size_t)row * N + col] = acc;
}

__global__ void init_mat(float* M, int N, float seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * N) {
        M[i] = (float)((i % 17) + 1) * 0.0625f + seed;
    }
}

// v2：共享内存 tiling + 每线程 4x4 寄存器分块
// 关键：每个 k 步从共享内存取 4+4 个值，做 16 个 FMA -> 取数:FMA 从 2:1 降到 0.5:1
#define TS 16
#define BM 64
#define BN 64
#define TM 4
#define TN 4

__global__ void sgemm_reg(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int N) {
    __shared__ float As[TS][BM + 4];
    __shared__ float Bs[TS][BN + 4];

    int tx = threadIdx.x;              // 0..15
    int ty = threadIdx.y;              // 0..15
    int tid = ty * 16 + tx;
    int rowStart = blockIdx.y * BM;
    int colStart = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int t = 0; t < N; t += TS) {
        for (int l = 0; l < (BM * TS) / 256; ++l) {
            int idx = tid + l * 256;
            As[idx / BM][idx % BM] = A[(size_t)(rowStart + idx % BM) * N + t + idx / BM];
        }
        for (int l = 0; l < (TS * BN) / 256; ++l) {
            int idx = tid + l * 256;
            Bs[idx / BN][idx % BN] = B[(size_t)(t + idx / BN) * N + colStart + idx % BN];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TS; ++k) {
            float aReg[TM], bReg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(size_t)(rowStart + ty * TM + i) * N + colStart + tx * TN + j] = acc[i][j];
}

// v3：按官方 rules 的三条指路修
//   1) A 的 tile 装载原来跨行访问（warp 内 32 个线程读 32 行，步长 N）
//      改成连续线程读连续的 k -> 合并
//   2) C 的写回原来按 tx*TN 跨步（warp 内 stride=4）-> 改成 float4 一次写 16 B
//   3) 共享内存取数改成 float4 -> 减少 MIO 指令数
__global__ void sgemm_reg2(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N) {
    __shared__ float As[TS][BM + 4];
    __shared__ float Bs[TS][BN + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;
    int rowStart = blockIdx.y * BM;
    int colStart = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int t = 0; t < N; t += TS) {
        // A tile：连续线程读连续的 k（原来这里是跨行的）
        for (int l = 0; l < (BM * TS) / 256; ++l) {
            int idx = tid + l * 256;
            int r = idx / TS;
            int c = idx % TS;
            As[c][r] = A[(size_t)(rowStart + r) * N + t + c];
        }
        for (int l = 0; l < (TS * BN) / 256; ++l) {
            int idx = tid + l * 256;
            Bs[idx / BN][idx % BN] = B[(size_t)(t + idx / BN) * N + colStart + idx % BN];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TS; ++k) {
            float4 a4 = *reinterpret_cast<const float4*>(&As[k][ty * TM]);
            float4 b4 = *reinterpret_cast<const float4*>(&Bs[k][tx * TN]);
            float aReg[TM] = {a4.x, a4.y, a4.z, a4.w};
            float bReg[TN] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

    // C 写回：float4 一次 16 B（原来 stride=4，每 sector 只用 8/32 字节）
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        float4 v = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(
            &C[(size_t)(rowStart + ty * TM + i) * N + colStart + tx * TN]) = v;
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = (size_t)N * N * sizeof(float);
    const double flops = 2.0 * (double)N * N * N;

    float *A, *B, *C0, *C1;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C0, bytes));
    CUDA_CHECK(cudaMalloc(&C1, bytes));
    init_mat<<<(N * N + 255) / 256, 256>>>(A, N, 0.0f);
    init_mat<<<(N * N + 255) / 256, 256>>>(B, N, 1.0f);
    CUDA_CHECK(cudaGetLastError());

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    print_header("sgemm N=1024  (working set 12 MB, L2 is only 1 MB)");
    printf("  %-16s %12s %12s %14s\n", "version", "ms", "GFLOP/s", "pct of peak");

    float ms0 = bench_ms([&] { sgemm_naive<<<grid, block>>>(A, B, C0, N); }, 3, 10);
    printf("  %-16s %12.3f %12.1f %13.1f%%\n", "naive", ms0,
           flops / (ms0 * 1e-3) / 1e9, flops / (ms0 * 1e-3) / 1e9 / 2760.0 * 100.0);

    float ms1 = bench_ms([&] { sgemm_tiled<<<grid, block>>>(A, B, C1, N); }, 3, 10);
    printf("  %-16s %12.3f %12.1f %13.1f%%\n", "tiled(32x32)", ms1,
           flops / (ms1 * 1e-3) / 1e9, flops / (ms1 * 1e-3) / 1e9 / 2760.0 * 100.0);
    printf("  speedup          %.2fx\n", ms0 / ms1);

    // v2：tiling + 4x4 寄存器分块
    float* C2;
    CUDA_CHECK(cudaMalloc(&C2, bytes));
    dim3 block2(16, 16);
    dim3 grid2(N / BM, N / BN);
    float ms2 = bench_ms([&] { sgemm_reg<<<grid2, block2>>>(A, B, C2, N); }, 3, 10);
    printf("  %-16s %12.3f %12.1f %13.1f%%\n", "tiled+reg(4x4)", ms2,
           flops / (ms2 * 1e-3) / 1e9, flops / (ms2 * 1e-3) / 1e9 / 2760.0 * 100.0);
    printf("  speedup vs naive %.2fx\n", ms0 / ms2);

    // v3：按官方 rules 修访存模式 + 向量化
    float* C3;
    CUDA_CHECK(cudaMalloc(&C3, bytes));
    float ms3 = bench_ms([&] { sgemm_reg2<<<grid2, block2>>>(A, B, C3, N); }, 3, 10);
    printf("  %-16s %12.3f %12.1f %13.1f%%\n", "reg2(+vectorized)", ms3,
           flops / (ms3 * 1e-3) / 1e9, flops / (ms3 * 1e-3) / 1e9 / 2760.0 * 100.0);
    printf("  speedup vs naive %.2fx  (vs reg %.2fx)\n", ms0 / ms3, ms2 / ms3);

    // 正确性：两个版本 + 抽查
    print_header("correctness");
    float* h0 = (float*)malloc(bytes);
    float* h1 = (float*)malloc(bytes);
    float* h2 = (float*)malloc(bytes);
    float* h3 = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h0, C0, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h1, C1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2, C2, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h3, C3, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    float worst = 0.f;
    int bad2 = 0;
    float worst2 = 0.f;
    int bad3 = 0;
    float worst3 = 0.f;
    for (int i = 0; i < N * N; ++i) {
        float d = fabsf(h0[i] - h1[i]);
        if (d > worst) worst = d;
        if (d > 1e-2f) ++bad;
        float d2 = fabsf(h0[i] - h2[i]);
        if (d2 > worst2) worst2 = d2;
        if (d2 > 1e-2f) ++bad2;
        float d3 = fabsf(h0[i] - h3[i]);
        if (d3 > worst3) worst3 = d3;
        if (d3 > 1e-2f) ++bad3;
    }
    printf("  tiled vs naive : %d elements, mismatches %d, max diff %.3e -> %s\n",
           N * N, bad, worst, bad == 0 ? "PASS" : "FAIL");
    printf("  reg   vs naive : %d elements, mismatches %d, max diff %.3e -> %s\n",
           N * N, bad2, worst2, bad2 == 0 ? "PASS" : "FAIL");
    printf("  reg2  vs naive : %d elements, mismatches %d, max diff %.3e -> %s\n",
           N * N, bad3, worst3, bad3 == 0 ? "PASS" : "FAIL");

    free(h0);
    free(h1);
    free(h2);
    free(h3);
    cudaFree(A); cudaFree(B); cudaFree(C0); cudaFree(C1); cudaFree(C2); cudaFree(C3);
    return 0;
}
