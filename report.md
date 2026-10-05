# FUSED W4A16 GEMV 算子优化
## 首先我需要理解一下题目
自回归生成就是模型每次预测一个Token，把他接到已有文本后面，再根据这些文本预测下一个。  
每次生成一个Token，都需要读取一整行的权重来计算一个输出，一共N个输出的概率，就需要把整个矩阵遍历一遍  
然后题目说“先反量化为 FP16 矩阵写回显存、再调用 GEMV”这句话我竟然没看懂，我去查了一下。**权重原本是INT4存在显存，这里需要先反量化，也就是一个kernel按照(q - Z) * S把整张权重转换成FP16，把结果写入显存，形成一张完整的FP16权重矩阵。GEMV是矩阵与向量相乘，也就是另一个Kernel再从显存读取这张FP16矩阵，与输入向量X相乘**  
我原本还想着为什么说推理计算速度慢，其实是因为要搬运的数据太多。一个INT4是0.5字节，然后这里需要读入0.5字节，写出2字节的FP16，再读入这2字节(**说到这里，我觉得应该有办法消除掉这个写出和读入，直接合并起来**)  
还有这个**FUSED W4A16 GEMV 算子**是什么意思呢？我又去查了一下。W4代表Weight用INT4存储，A16的A是Activation，也就是参与计算的输入用FP16存储，在这里就是向量X，GEMV就是矩阵与向量相乘，在这里就是Y= XWT。Fused意味着读入INT4权重后立即反量化并且乘加，直接得到Y，省去中间的FP16权重矩阵的写回和读取  
看到这里，输入向量X是float16类型，形状是(1, K)  
量化权重W是uint8类型，形状是(N, K)，不过因为是INT4，所以变成了K/2  
所以这里给了一个W[n, 2k] = W[n, k] & 0xF和W[n, 2k + 1] = (W[n, k] >> 4) & 0xF,用掩码获得不同的位数  
这里我一开始不理解为什么要有一个G，原来是因为不同区段的权重，数值范围可能差别很大，如果大家共用一对的S和Z，容易损失精度。比如前面128个都在0到1，后面128个在0到100，那差别就很大了。这里区分开来，每128个元素共用一个S和Z  
在这里，python传入tensor，C++封装启动CUDA Kernel，并把结果Tensor返回给Python。Pybind11让python调用这个C++函数  
了解到我们需要写一个kernel然后让C++调用，然后这个C++函数应该要被python调用（python真是个胶水语言啊）  
那么这几个文件的各自的分工就比较清晰了。**gemv_w4a16.cu负责写GPU kernel，还有普通的C++封装函数，这个封装函数要读Tensor，分配输出，启动算子，返回输出的tensor，也就是调度的工作。那个.cuh的就是个头文件，声明封装函数。binding.cpp负责用pybind11把C++函数暴露给python，让python能够调用gemv_w4a16()这个函数。然后setup.py是用来编译构建脚本的。他把.cpp还有.cu编译成python可以import的模块。test_local.py是测试脚本，生成测试输入，计算FP32参考结果，检查输出是否正确。**  
我第一个写的是算子，也就是gemv_w4a16.cu的算子。这里先写声明吧。  
看到题目给的是
```
def gemv_w4a16(
    x: torch.Tensor,        # [1, K],                 float16
    w_q: torch.Tensor,      # [N, K // 2],            uint8
    scales: torch.Tensor,   # [N, K // group_size],   float16
    zeros: torch.Tensor,    # [N, K // group_size],   float16
    group_size: int = 128) -> torch.Tensor:          # [1, N],  float16
```

意味着**输入的x是一个fp16类型，w_q是一个uint8_t类型，scales是一个fp16类型，zeros是一个fp16类型，group_size是int类型，最后的输出要是一个fp16类型。**   
那么我们就可以确定好这个算子怎么声明了。fp16就是half类型，所以我们就这样声明我们的算子。
```
__global__ void gemv_kernel(const half *x, const u_int8_t *w_q, const half *scales, const half *zeros, half *y, int K, int N, int group_size);
```
这里面的x,w_q,scales,zeros都是对应的输入参数，y则是输出。
**我个人感觉还是需要把这些参数写清楚的，特别是形状**    
x是输入的向量，有K个FP16，形状是(1, K)  
w_q是打包好的量化权重，形状是(N, K / 2),一个字节有2个INT4  
scales是每组的缩放系数，形状是(N, K / group_size)  
zeros是每组的零点Z，形状是(N, K / group_size)  
y是输出向量，包含N个FP16，形状是(1, N)  
K是输入向量的长度  
N是权重的行数，也就是输出的长度  
group_size是分组后一组有多少个，题目给的是128  
我这里打算让一个thread负责一个完整的Y[n].遍历权重的第n行，逐个解包，反量化，再与对应的x[k]相乘，累加到自己的sum里面，遍历完k个权重，然后转成FP16写入Y[n]  
下面是公式  
$$
Y[n] = \sum_{k=0}^{K-1}
X[k]\,
\left(q_{n,k} - Z[n,\lfloor k/G \rfloor]\right)\,
S[n,\lfloor k/G \rfloor]
$$

根据这个公式，我们很容易就可以写出算子来  
```
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
```

接下来完整写一下这个函数，我们看到，题目说python接口要求是  
```
def gemv_w4a16(
    x: torch.Tensor,        # [1, K],                 float16
    w_q: torch.Tensor,      # [N, K // 2],            uint8
    scales: torch.Tensor,   # [N, K // group_size],   float16
    zeros: torch.Tensor,    # [N, K // group_size],   float16
    group_size: int = 128) -> torch.Tensor:          # [1, N],  float16
```

那么我们接收的就得是torch.Tensor类型的参数，而且返回的也是torch.Tensor类型的。这一步真的是很麻烦，我一开始写的是这样的  

```
torch::Tensor gemv_w4a16(
    const torch::Tensor &x,
    const torch::Tensor &w_q,
    const torch::Tensor &scales,
    const torch::Tensor &zeros,
    int group_size
) {
    // 根据x的外形得出长度K
    int K = x.size(1);
    // 根据w_q的外形得出长度N
    int N = w_q.size(0);

    half *y;
    cudaMalloc(&y, N * sizeof(half));

    int threads = 128;
    int blocks = (N + threads - 1) / threads;

    gemv_kernel<<<blocks, threads>>>(x, w_q, scales, zeros, y, K, N, group_size);
    return y;
}

```

后来发现问题非常大，接收的是Tensor类型，就不能直接这样传送给算子kernel了，还得转换，查了一下，转换是这样写的  
```
const half *x_ptr = reinterpret_cast<const half *>(x.data_ptr<at::Half>());
```
看起来真费劲了，我们需要用这种方法把x,w_q,scales,zeros,y都转换了  


现在还有一个问题，注意到题目说**Kernel 必须在 PyTorch 当前流异步执行**，那也就是说，启动kernel的时候还要指定pytorch当前流。  
我学习了一下CUDA流相关的知识。  
CUDA流可以理解成GPU的任务队列，同一条流里的操作按顺序执行。**需要把Kernel接到pytorch当前队列里，才能与前后的pytorch操作保持顺序**，这一点非常重要！  
下面这一条是把当前CUDA设备切换到x所在的GPU，离开函数时自动恢复原设备  
```
c10::cuda::CUDAGuard guard(x.device());
```
这一条是取得当前设备上Pytorch正在使用的流  
```
cudaStream_t stream = at::cuda::getCurrentCUDAStream();
```
这一条是把kernel提交到这条流
```
gemv_kernel<<<blocks, threads, 0, stream>>>(...);
```
下面这一条是检查kernel启动是否出错
```
C10_CUDA_KERNEL_LAUNCH_CHECK();
```

最后我们写好这个，应该是这样的  
```
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
```


现在我们的gemv_w4a16.cu和gemv_w4a16.cuh都写好了，我们继续写binding.cpp  

## pybind部分
这里就有必要学习一点pybind的基础知识了。pybind是一个C++库，他的作用就是让python能调用写好的C++函数。比如在C++写了一个add函数，通过pybind注册后，python就可以写模块.add(2, 3)这样了。在调用的时候pybind在中间做转换，python的整数换成C++的int，然后调用C++函数，返回值再转换成python整数。  
在C++中写m.def("add", &add)就是告诉pybind，python模块里的"add"对应这个C++函数add。  
**对于现在这道题有几个东西要知道的。**  
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)代表python在import这个模块的时候，执行里面的注册代码，TORCH_EXTENSION_NAME代表由构建工具根据setup.py中的模块名，m代表模块  
m.def("NAME", &NAME,...)前面是python中的函数名，后面是C++的函数指针，通过这个把两者连接起来  
pybind11::arg(...)是指定python参数名称，也允许用关键字传参。  

那么我们就可以写binding.cpp了  
```
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
```
m.def连接python的函数名和C++函数，pybind11::arg就是指定python的参数名，并且这里设置group_size就是128  

接下来还需要写setup.py，这是一个构建脚本，负责组织C++和CUDA源码的编译链接，生成python能够导入的拓展模块  
实话说这个我真的不会写，只好借助AI了。这里他给的是  
```
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_w4a16_gemv",
    ext_modules=[
        CUDAExtension(
            name="custom_w4a16_gemv",
            sources=[
                "csrc/binding.cpp",
                "csrc/gemv_w4a16.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

我理解了一下源代码，首先导入的这几个工具，setup负责接收构建配置，处理命令行构建指令
CUDAExtension负责描述要构建的C++/CUDA拓展模块，补充需要的头文件路径和链接库
BuildExtension就是根据这些配置组织实际的编译和链接  
name就是发行包的名称，ext_modules是扩展模块列表，我们这里只有一个模块，所以只有这个CUDAExtension  
然后name还是扩展模块名称，决定了python如何导入，比如这里我们就可以写  
```
import custom_w4a16_gemv
```
sources这里指定源码，这些路径是相对于运行构建命令的目录  
后面这些都是编译参数了  
```
extra_compile_args={
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
    ],
}
```

其中-O3就是开O3优化，-U__CUDA_NO_HALF_OPERATORS__就是允许使用half的加法等运算，-U__CUDA_NO_HALF_CONVERSIONS__就是取消禁用half隐式转换的宏，允许half和float转换  
最后这句cmdclass={"build_ext": BuildExtension}就是说遇到build_ext构建指令的时候使用pytorch提供的BuildExtension来执行  
然后构建的时候我们就可以用  
```
python setup.py build_ext --inplace
```
就是运行构建脚本，构建这个拓展模块，然后把最后生成的.so放在项目的目录，这样就可以直接import了  

运行后得到了一个custom_w4a16_gemv.cpython-310-x86_64-linux-gnu.so文件，这是一个动态链接库。
后面在python执行import的时候就可以直接加载了  

现在我们只剩下test_local.py了。这个函数负责按照题目要求，生成输入
也就是按照题目的要求CUDA上的x，打包权重w_q，scales和zeros；然后调用算子，计算参考答案，比较结果  
这一部分似乎不是必须的，那我还是交给AI生成了hhh
```
"""W4A16 GEMV 本地正确性测试。

把本文件放在 setup.py 和编译生成的 .so 所在目录，执行：
    python local_test.py

Python 反量化仅用于构造独立的 FP32 参考结果。
测试不调用 torch.mm，也不显式同步 CUDA 设备。
"""

import argparse

import torch
import torch.nn.functional as F

import custom_w4a16_gemv


GROUP_SIZE = 128

# 每项为 (K, N)。小形状覆盖单行、多组和 N 不能被 128 整除的情况。
CASES = [
    (128, 1),
    (256, 5),
    (384, 129),
    (640, 257),
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
]


def make_inputs(k, n, device, seed):
    """生成题目要求的四个输入；每组的 S、Z 都可以不同。"""
    torch.manual_seed(seed)
    x = torch.randn((1, k), device=device, dtype=torch.float16)

    # 随机字节的低 4 位、高 4 位分别代表两个 0..15 的量化值。
    w_q = torch.randint(
        0, 256, (n, k // 2), device=device, dtype=torch.uint8
    )

    groups = k // GROUP_SIZE
    scales = (
        torch.rand((n, groups), device=device, dtype=torch.float32)
        * 0.045 + 0.005
    ).to(torch.float16)

    # Z 按题意是 FP16，因此也测试包含小数的零点。
    zeros = (
        torch.rand((n, groups), device=device, dtype=torch.float32) * 15.0
    ).to(torch.float16)

    return x, w_q, scales, zeros


def reference_fp32(x, w_q, scales, zeros):
    """独立参考：解包 -> FP32 反量化 -> 逐元素乘法并按 K 求和。"""
    n = w_q.size(0)
    k = x.size(1)

    # 恢复 [N, K] 的逻辑量化矩阵，偶数列用低半字节。
    q = torch.empty((n, k), device=w_q.device, dtype=torch.float32)
    q[:, 0::2] = (w_q & 0x0F).float()
    q[:, 1::2] = ((w_q >> 4) & 0x0F).float()

    # [N, K/G, G] 让每组的 128 个权重共用同一个 S、Z。
    q_grouped = q.reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    weights = (
        (q_grouped - zeros.float().unsqueeze(-1))
        * scales.float().unsqueeze(-1)
    ).reshape(n, k)

    # x.float() 的 [1, K] 会广播到每一行权重。
    return (weights * x.float()).sum(dim=1).unsqueeze(0)


def run_case(k, n, device, seed, label):
    x, w_q, scales, zeros = make_inputs(k, n, device, seed)

    y_pred = custom_w4a16_gemv.gemv_w4a16(
        x, w_q, scales, zeros, group_size=GROUP_SIZE
    )
    y_ref = reference_fp32(x, w_q, scales, zeros)

    assert isinstance(y_pred, torch.Tensor), "输出必须是 Tensor"
    assert y_pred.shape == (1, n), f"输出形状错误：{y_pred.shape}"
    assert y_pred.dtype == torch.float16, f"输出类型错误：{y_pred.dtype}"
    assert y_pred.device == x.device, f"输出设备错误：{y_pred.device}"
    assert y_pred.is_contiguous(), "输出必须 Contiguous"

    pred_float = y_pred.float()
    error = (pred_float - y_ref).abs()
    cosine = F.cosine_similarity(pred_float, y_ref, dim=-1).mean().item()

    # .item() 会等待相关结果可读；不需要 cudaDeviceSynchronize。
    print(
        f"[{label}] K={k}, N={n}, seed={seed}: "
        f"max_abs={error.max().item():.6g}, "
        f"mean_abs={error.mean().item():.6g}, cosine={cosine:.8f}",
        flush=True,
    )

    # 严格使用题目的两项精度门槛，不因当前 Kernel 而放宽。
    torch.testing.assert_close(
        pred_float, y_ref, rtol=1e-2, atol=1e-2
    )
    assert cosine > 0.999, f"余弦相似度未达标：{cosine}"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="W4A16 GEMV 正确性测试")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("本测试需要支持当前 PyTorch 的 CUDA GPU")

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device 必须指定 CUDA 设备，例如 cuda:0")

    failed = 0
    total = len(CASES) + 1
    with torch.cuda.device(device):
        for index, (k, n) in enumerate(CASES):
            try:
                run_case(k, n, device, args.seed + index, "current stream")
            except AssertionError as error:
                failed += 1
                print(f"FAIL: {error}", flush=True)
            else:
                print("PASS", flush=True)

        # 在非默认流上生成输入并调用算子，检查当前流的基本使用。
        # 随机数据操作与算子应进入同一条流；无需显式设备同步。
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            try:
                run_case(4096, 129, device, args.seed + len(CASES),
                         "non-default stream")
            except AssertionError as error:
                failed += 1
                print(f"FAIL: {error}", flush=True)
            else:
                print("PASS", flush=True)

    print(f"\n结果：{total - failed}/{total} 通过", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

```