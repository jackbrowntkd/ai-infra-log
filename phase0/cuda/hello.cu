#include <cstdio>
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    int n = 1 << 10;
    printf("compiled and ran, n=%d\n", n);
    return 0;
}