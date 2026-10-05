#include <cuda_fp16.h>
#include <stdint.h>
__global__ void gemv_kernel(
    const half *x, 
    const u_int8_t *w_q, 
    const half *scales, 
    const half *zeros, 
    half *y, 
    int K, int N, 
    int group_size);