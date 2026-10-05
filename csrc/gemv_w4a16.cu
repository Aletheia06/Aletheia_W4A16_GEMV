#include <cuda_runtime.h>
#include <cuda_fp16.h>
__global__ void gemv_kernel(const half *x, const u_int8_t *w_q, const half *scales, const half *zeros, half *y, int K, int N, int group_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // 一个thread负责计算一个Y[i]
    if(i >= N) {
        return;
    }

    half sum = 0.0f;
    for(int k = 0;k < K;k++) {
        // 因为这里的w_q是打包的
        // 所以一次拿出来的是2个
        const u_int8_t w_q_of_2 = w_q[i * K/2 + k/2];
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