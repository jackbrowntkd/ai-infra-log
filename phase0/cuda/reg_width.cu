// 阶段 0 · 验证寄存器格子有多宽
//
// 结论：通用寄存器是 32 位一格。
//   float32 / int32  -> 1 格
//   double / 指针     -> 2 格
//   half2（两个 fp16） -> 1 格
//
// 用两条独立证据来验：
//   证据1 直接看 PTX：mov.b64 把一个 64 位数拆进两个 32 位寄存器
//   证据2 看寄存器用量：同样"养着 K 个值"，double 是 float 的两倍

#include <cstdio>
#include <cuda_runtime.h>

// ---------- 证据 1 ----------
// mov.b64 {%0,%1}, %2  这条 PTX 指令的字面意思就是
// "把一个 64 位值拆到底两个 32 位寄存器里"
__global__ void split64(unsigned int* out) {
    unsigned long long v = 0x1122334455667788ULL;
    unsigned int lo, hi;
    asm("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(v));
    out[0] = lo;
    out[1] = hi;
}

// ---------- 证据 2 ----------
// 让 K 个值两两相乘，逼它们同时活着（否则编译器会把加载推迟）
template <int K, typename T>
__global__ void pairwise(const T* in, T* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T v[K];
#pragma unroll
        for (int k = 0; k < K; ++k) v[k] = in[(size_t)i * K + k];

        T s = (T)0;
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = k + 1; j < K; ++j) s += v[k] * v[j];

        out[i] = s;
    }
}

void force_instantiation() {
    pairwise<8, float><<<1, 1>>>(nullptr, nullptr, 0);
    pairwise<8, double><<<1, 1>>>(nullptr, nullptr, 0);
}

int main() {
    unsigned int* d;
    cudaMalloc(&d, 2 * sizeof(unsigned int));
    split64<<<1, 1>>>(d);

    unsigned int h[2];
    cudaMemcpy(h, d, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("原始 64 位值 : 0x1122334455667788\n");
    printf("拆出的两个 32 位寄存器:\n");
    printf("  hi = 0x%08X\n", h[1]);
    printf("  lo = 0x%08X\n", h[0]);
    printf("拼回去       : 0x%08X%08X\n", h[1], h[0]);
    printf("\n-> 一个 64 位值确实占两个 32 位寄存器\n");

    cudaFree(d);
    return 0;
}