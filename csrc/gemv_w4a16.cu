#include <cuda_runtime.h>
#include "gemv_w4a16.cuh"
__global__ void gemv_kernel(const half *x, const uint8_t *w_q, const half *scales, const half *zeros, half *y, int K, int N, int group_size) {
    // 现在计算当前的线程再warp中的编号
    int lane = threadIdx.x % 32;
    // 然后计算当前warp在block内的编号
    int warp_id = threadIdx.x / 32;
    // 计算每个block有多少个warp
    int warps_per_block = blockDim.x / 32;
    // 同一个warp内的32个线程计算同一行i
    // 这里的i就是当前warp的全局编号
    // 所以也相当于行号了
    int i = blockIdx.x * warps_per_block + warp_id;
    if(i >= N) {
        return;
    }

    // 这个row代表当前第i行权重的起始地址
    // 一行有K / 2个
    const uint8_t *row = w_q + i * K / 2;
    float sum = 0.0f;
    // 下面的k代表包的编号
    // 每个包有8个权重，总共K/8个包
    // 一轮完成32个包，这样lane0负责0,32,64包，lane1是1，33,65包
    for(int k = lane;k < K / 8;k += 32) {
        // packed就是把四个字节合并
        uint32_t packed = reinterpret_cast<const uint32_t *>(row)[k];

        // group_index是就是计算当前到第几个组了
        // 用的是当前到第几个权重，除以一组多少个权重来计算的
        int group_index = i * K / group_size + k * 8 / group_size;


        // 根据组号得到z和s
        // 顺便补一下之前忘记的转换
        float z = __half2float(zeros[group_index]);
        float s = __half2float(scales[group_index]);

        // 这里的m代表当前包内的第m个INT4权重，范围是0-7，因为一个包就是8个INT4
        // 然后我们用按位与
        // 把4位提取出来，给q
        for(int m = 0; m < 8; m++) {
            int q = (packed >> (m * 4)) & 0xF;

            float xv = __half2float(x[k * 8 + m]);

            float weight = (q - z) * s;
            sum += xv * weight;
        }
    }

    // 现在每个线程的sum都是自己的lane和32结合的产物
    // 也就是这个sum就是第lane, lane + 32, lane + 64 ...的和
    // 所以接下来合并
    for(int j = 16;j > 0;j /= 2) {
        // 这一行就是共享大家（warp内）的sum
        // 然后每个线程都和比自己大16的加起来
        // 变成第0个和第16个合并，第一个和第17个合并...
        // 后面只剩下0到15
        // 然后继续按照8为步长合并
        // 以此类推一直到1
        sum += __shfl_down_sync(0xffffffffu, sum, j);
    }

    // 最后大家的都加到了第0个上去
    if(lane == 0) {
        y[i] = __float2half_rn(sum);
    }
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
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;

    // 在pytorch当前流启动kernel
    // 先指定stream为当前的pytorch流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 然后启动kernel！！！
    gemv_kernel<<<blocks, threads, 0, stream>>>(x_ptr, w_ptr, s_ptr, z_ptr, y_ptr, K, N, group_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
