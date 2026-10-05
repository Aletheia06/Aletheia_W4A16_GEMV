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