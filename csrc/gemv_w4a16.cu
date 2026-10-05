#include <cuda_runtime.h>
#include "gemv_w4a16.cuh"
__global__ void gemv_kernel(const half *x, const uint8_t *w_q, const half *scales, const half *zeros, half *y, int K, int N, int group_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // 一个thread负责计算一个Y[i]
    if(i >= N) {
        return;
    }

    half sum = 0.0f;
    for(int k = 0;k < K;k++) {
        // 因为这里的w_q是打包的
        // 所以一次拿出来的是2个
        const uint8_t w_q_of_2 = w_q[i * K/2 + k/2];
        // 需要根据奇数偶数来判断拿哪个
        // q代表现在要用的权重
        int q = 0;
        if(k % 2 == 0) {
            q = w_q_of_2 & 0xF;
        }
        else {
            q = (w_q_of_2 >> 4) & 0xF;
        }

        // 现在还需要知道当前的权重属于哪个组
        int group_index = i * K / group_size + k / group_size;

        // 根据组号得到z和s
        float z = zeros[group_index];
        float s = scales[group_index];
        // 拿到现在的x
        float xv = x[k];
        
        float weight = (q - z) * s;
        sum += xv * weight;
    }
    y[i] = sum;
}

torch::Tensor gemv_w4a16(
    const torch::Tensor &x,
    const torch::Tensor &w_q,
    const torch::Tensor &scales,
    const torch::Tensor &zeros,
    int group_size
) {

    // 使用x所在的CUDA设备
    c10::cuda::CUDAGuard guard(x.device());

    // 根据x的外形得出长度K
    int K = x.size(1);
    // 根据w_q的外形得出长度N
    int N = w_q.size(0);

    // 因为我们需要的返回值是torch类型的
    // 所以这里需要在这个设备上分配FP16输出，形状是(1, N)
    auto y = torch::empty({1, N}, x.options());

    // 后面这几步全部都是为了把Tensor转成显存指针
    // 让kernel能用
    const half *x_ptr = reinterpret_cast<const half *>(x.data_ptr<at::Half>());
    const uint8_t *w_ptr = w_q.data_ptr<uint8_t>();
    const half *s_ptr = reinterpret_cast<const half *>(scales.data_ptr<at::Half>());
    const half *z_ptr = reinterpret_cast<const half *>(zeros.data_ptr<at::Half>());
    half *y_ptr = reinterpret_cast<half *>(y.data_ptr<at::Half>());

    // 给定一个block有128个threads
    int threads = 128;
    int blocks = (N + threads - 1) / threads;

    // 在pytorch当前流启动kernel
    // 先指定stream为当前的pytorch流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 然后启动kernel！！！
    gemv_kernel<<<blocks, threads, 0, stream>>>(x_ptr, w_ptr, s_ptr, z_ptr, y_ptr, K, N, group_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
