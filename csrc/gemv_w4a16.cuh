#include <cuda_fp16.h>
#include <stdint.h>

// 下面这个头文件是为了torch::Tensor的
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


__global__ void gemv_kernel(
    const half *x, 
    const u_int8_t *w_q, 
    const half *scales, 
    const half *zeros, 
    half *y, 
    int K, int N, 
    int group_size);