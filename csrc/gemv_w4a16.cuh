#include <cuda_fp16.h>
#include <stdint.h>

// 下面这个头文件是为了torch::Tensor的
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


torch::Tensor gemv_w4a16(
    const torch::Tensor &x,
    const torch::Tensor &w_q,
    const torch::Tensor &scales,
    const torch::Tensor &zeros,
    int group_size
);