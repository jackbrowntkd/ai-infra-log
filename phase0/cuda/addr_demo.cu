// What is an "address"?
//
// Claim: an address is a LABEL (a number), not a position inside VRAM.
// The address space is far larger than the physical memory.
//
// HARD RULE for this project: string literals in .cu files must be ASCII only.
// nvcc mishandles non-ASCII in string literals -- sometimes it compiles,
// sometimes it reports "missing closing quote", sometimes it eats the
// adjacent escape sequence. Do not fight it. Chinese goes in comments.
//
// How to say:  .\phase0\cuda\addr_demo.exe

#include <cstdio>
#include <cuda_runtime.h>

static void nl() { putchar('\n'); }

static int top_bit(unsigned long long v) {
    int b = -1;
    while (v) { v >>= 1; ++b; }
    return b;
}

int main() {
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);

    printf("VRAM total          : %.2f GB", totalB / 1073741824.0); nl();
    printf("pointer width       : %zu bits", sizeof(void*) * 8); nl();
    nl();

    void* p[4];
    for (int i = 0; i < 4; ++i) cudaMalloc(&p[i], 1024);

    printf("addresses from 4x cudaMalloc(1024):"); nl();
    for (int i = 0; i < 4; ++i) {
        unsigned long long a = (unsigned long long)p[i];
        printf("  p%d = 0x%012llX  = %-14llu (%.1f GB)", i, a, a, a / 1e9); nl();
    }
    nl();

    unsigned long long a0 = (unsigned long long)p[0];
    int bits = top_bit(a0) + 1;

    printf("--- the point ---"); nl();
    printf("physical VRAM       : %.1f GB", totalB / 1e9); nl();
    printf("but address value is: %.1f GB   -> %.0fx larger than VRAM",
           a0 / 1e9, a0 / (double)totalB); nl();
    printf("bits needed         : %d  (32-bit addressing maxes out at 4 GB)", bits); nl();
    printf("64-bit space limit  : 16 EB (18446744073 GB)"); nl();
    nl();
    printf("=> an address is a LABEL, not a place. Big label range != big memory."); nl();

    for (int i = 0; i < 4; ++i) cudaFree(p[i]);
    return 0;
}