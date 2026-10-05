#include <torch/extension.h>
#include "gemv_w4a16.cuh"

PYBIND11_MODULE(gemv_w4a16, m) {
    m.def(
        "gemv_w4a16",
        &gemv_w4a16,
        pybind11::arg("x"),
        pybind11::arg("w_q"),
        pybind11::arg("scales"),
        pybind11::arg("zeros"),
        pybind11::arg("group_size") = 128
    );
}

torch::Tensor gemv_w4a16(
    const torch::Tensor& x,
    const torch::Tensor& w_q,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int group_size = 128
);

