#include <torch/extension.h>

torch::Tensor gemv_w4a16(
    const torch::Tensor& x,
    const torch::Tensor& w_q,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int group_size = 128
);

