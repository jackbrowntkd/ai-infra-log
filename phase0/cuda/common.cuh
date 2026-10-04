#pragma once
// 阶段 0 共用的 kernel 测试脚手架。
//
// 计时方法刻意和 phase0/bench.py 对齐：
//   预热若干次 -> CUDA event 卡住整段循环 -> 除以次数
//
// 唯一区别：这里不刷 L2。因为下面每个 kernel 的访存工作集都远大于 L2
// （本机 L2 只有 1 MB），数据不可能整块留在 cache 里，不刷也不影响结论。
// 以后要测小 size 的 kernel 时，记得回头检查这个假设。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s -> %s\n", __FILE__,        \
                    __LINE__, #call, cudaGetErrorString(err_));              \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// 返回单次调用的平均耗时（毫秒）
template <typename F>
float bench_ms(F&& fn, int warmup = 10, int rep = 50) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < rep; ++i) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / rep;
}

inline void print_header(const char* title) {
    printf("\n%s\n", title);
    for (size_t i = 0; i < strlen(title); ++i) putchar('-');
    putchar('\n');
}

// 一个 (算术强度, 实测性能) 点，给 phase0/roofline.py 画图用
struct Point {
    const char* name;
    double ms;
    double gflops;  // 不适用填 0
    double gbps;    // 不适用填 0
    double ai;      // FLOP/byte
};

inline void write_points(const char* path, const Point* pts, int n) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "警告：写不了 %s\n", path);
        return;
    }
    fprintf(f, "{\n  \"points\": [\n");
    for (int i = 0; i < n; ++i) {
        fprintf(f,
                "    {\"name\": \"%s\", \"ms\": %.6f, \"gflops\": %.4f, "
                "\"gbps\": %.2f, \"ai\": %.4f}%s\n",
                pts[i].name, pts[i].ms, pts[i].gflops, pts[i].gbps, pts[i].ai,
                (i + 1 < n) ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("\n已写入 %s\n", path);
}