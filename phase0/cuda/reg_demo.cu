// 用来观察寄存器"格子数"的小 kernel
//
// 一个指针  -> 需要 64 位 -> 占 2 格
// 一个 float4 -> 128 位     -> 占 4 格
// 规则是同一条：格数 = 向上取整(宽度 / 32)

#include <cstdio>
#include <cuda_fp16.h>

__global__ void demo(const float4* in, float4* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float4 v = in[i];
        out[i] = make_float4(v.x * 2.f, v.y * 2.f, v.z * 2.f, v.w * 2.f);
    }
}

int main() {
    printf("sizeof(void*)   = %zu 字节  (%zu 位)\n", sizeof(void*), sizeof(void*) * 8);
    printf("sizeof(float)   = %zu 字节\n", sizeof(float));
    printf("sizeof(double)  = %zu 字节\n", sizeof(double));
    printf("sizeof(float4)  = %zu 字节  (%zu 位)\n", sizeof(float4), sizeof(float4) * 8);
    printf("sizeof(half)    = %zu 字节\n", sizeof(__half));
    return 0;
}