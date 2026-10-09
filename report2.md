# FUSED W4A16 GEMV 算子优化

# 先学习一下怎么看Nvidia Nsight Compute
1.DRAM Throughput是显存数据搬运能力用了多少，衡量显存访问相关硬件的吞吐量，相对于其峰值达到了多少，衡量的是搬运速度；影响因素是有没有持续、充足的显存访问请求  
2.Compute(SM) Throughput是SM内最接近满负荷的处理环节用了多少。主要受到执行哪些指令，以及能否不断提供可执行的指令影响  
3.Theoretical Occupancy是已驻留的 warp 数占硬件最多能容纳的 warp 数的比例
4.Registers Per Thread就是编译器为每个线程分配的寄存器数量  
5.Block Limit Registers是只考虑寄存器容量，一个SM最多同时驻留这么多个这样的block。因为SM的寄存器总量有限，每个block分走一部分，分给这么多个，剩下的资源就不够放下一个完整的block了  
6.Waves Per SM是全部block的工作量相当于装满GPU多少轮.怎么计算的呢？比如GPU有26个SM，每个SM最多同时驻留12个block，所以一共可以同时312个block，如果启动了1024个block，那就是1024/312=3.28  
7.Duration是这一次kernel在GPU上执行了多久，时间缩短才是最终的性能收益，对于这道题，Duration是最重要的  
8.SM Frequency是测量周期内他的频率  
9.Elapsed Cycles是整个测量期间经过了多少个核心时钟周期，其实也就是把Duration转化成了周期，大概就是Duration*1530  
10.SM Active Cycles是平均每个SM有warp驻留的周期数。  
11.Memory Throughput是访存路径中，相关处理细节最接近峰值的程度，相比于DRAM Throughput，Memory Throughput的范围更广，还涉及缓存和处理访存请求的硬件环节。它从相关子指标中取最高百分比，不是把各级吞吐率相加或平均  
12.L1/TEX Cache Throughput是SM附近的缓存与访存处理单元有多接近处理能力上限  
13.L2 Cache Throughput是全GPU共享的二级缓存有多接近处理能力上限  
14.Share Memory Configuration Size是每个SM当前划给共享内存的容量  
15.Static Shared Memory Per Block和Dynamic Share Memory Per Block和Driver Shared Memory Per Block都是看每个block的共享内存用量的。
16.Total SM Elapsed Cycles是把所有SM经过的周期数加起来  
17.Average SMSP Active Cycles是平均每个SM子分区有warp驻留的周期数。SMSP是SM内部的一个执行分区，每个SM有4个SMSP，各自有warp调度器和执行资源。warp分配到其中一个分区执行。一个SMSP至少有一个warp驻留，就记为活跃  
18.Total SMSP Elapsed Cycles是把所有SM子分区经过的周期数加起来  
19.Average DRAM Active Cycles是平均每个显存计数单元进行数据传输的活跃周期数  
20.Total DRAM Elapsed Cycles是把所有显存计数单元经过的周期数加起来  
21.Block Limit SM是SM本身能管理的驻留block数量上限，由硬件决定  
22.Block Limit Barriers是同步屏障资源允许同时驻留多少block  
23.TPCs是GPU中参与这次执行的TPC硬件组数，可以理解为SM外面的一层硬件分组，包含一个或者多个SM
24.Stack Size是这次启动的时候每个GPU显存的调用栈大小  
25.THread Block Cluster是让多个block组成一个可以协作的组。等于0表示没有启用显式cluster配置  
26.Green Context式让某项CUDA工作使用指定的一部分GPU资源，等于0表示关闭  



# 提高带宽利用率  
在前面提交的那一份，已经实现了基础功能，可以准确计算结果了。接下来根据题目的要求，还得提高带宽利用率。  
要提高利用率，先看看哪些地方可以被提高的。  
查了一下，有下面这几个分析工具。第一个是CUDA Events，他测量GPU流中这一段执行用了多久；第二个是Nsight Compute，他来测量单个kernel使用了哪些硬件资源，哪里在等待；第三个是Nsight System，他测量CPU和GPU的时间线上，哪里有启动开销，空闲，或者等待  
**有效带宽利用率**是逻辑有效数据量除以kernel耗时*GPU峰值带宽。那么数据肯定是不变的，GPU的峰值带宽也是固定的。也就是，我们**能优化的只有时间了**  
耗时有这几个影响因素。一个是**实际搬运量**，也就是，我们可以去看看，**同样的计算，有没有产生重复读取或者浪费的访存事务**；第二个是**搬运的并行程度**，**就是能不能有持续的访存请求，让显存系统充分工作**；第三个是**处理数据的开销**，比如**解包，类型转换，乘加**这些  
其实我们能优化的也只有gemv_w4a16.cu这个文件了，因为其他几个都是基本固定了的。  

我们来使用ncu工具来测试一下现在的有效带宽和利用率。因为之前AI写的测试脚本中，前4个都是小尺寸用例，第五个才是题目的正式形状。所以需要skip前四个，测试第五个。另外我们还需要指定名字为gemv_kernel的算子。此外一开始报错了，所以加上第一句，允许跟踪子进程    
```
ncu \
  --target-processes all \
  --kernel-name regex:gemv_kernel \
  --launch-skip 4 \
  --launch-count 1 \
  --set basic \
  python test_local.py
```

运行，不知道为什么显示ERR_NVGPUCTRPERM  
说是ncu没有权限读GPU的性能计数器  
这里太折腾了，我还是用自己的电脑试一下吧。发现还有这个问题，于是查了一下，发现需要从“系统 → 高级 → 开发者 → 管理 GPU 性能计数器 → 允许所有用户访问”这里开启，然后就好了。  
很好，现在得到了这样的分析结果  
```
[14435] python3.10@127.0.0.1
  gemv_kernel(const __half *, const unsigned char *, const __half *, const__half *, __half *, int, int, int) (32, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.99
    SM Frequency                    Ghz         1.69
    Elapsed Cycles                cycle      1621955
    Memory Throughput                 %        60.92
    DRAM Throughput                   %         2.64
    Duration                         us       962.50
    L1/TEX Cache Throughput           %        91.13
    L2 Cache Throughput               %         1.11
    SM Active Cycles              cycle   1084265.23
    Compute (SM) Throughput           %        13.18
    ----------------------- ----------- ------------

    OPT   Memory is more heavily utilized than Compute: Look at the MemoryWorkload Analysis section to identify the L1 
          bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes      
          transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or        
          whether there are values you can (re)compute.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     32
    Preferred Cluster Size                                         0
    Registers Per Thread             register/thread              40
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              26
    Stack Size                                                  1024
    Threads                                   thread            4096
    # TPCs                                                        13
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.10
    -------------------------------- --------------- ---------------

    OPT   If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have at least two  
          blocks per multiprocessor (compared to the currently executed 1.2 blocks) This way, blocks that aren't        
          waiting for __syncthreads() can keep the hardware busy.                                             

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block           12
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        11.18
    Achieved Active Warps Per SM           warp         5.37
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 88.82%                                             
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (11.2%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                             

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       279032
    Total DRAM Elapsed Cycles        cycle     42311680
    Average L1 Active Cycles         cycle   1084265.23
    Total L1 Elapsed Cycles          cycle     42167630
    Average L2 Active Cycles         cycle    849579.12
    Total L2 Elapsed Cycles          cycle     25290560
    Average SM Active Cycles         cycle   1084265.23
    Total SM Elapsed Cycles          cycle     42167630
    Average SMSP Active Cycles       cycle   1082780.62
    Total SMSP Elapsed Cycles        cycle    168670520
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 22.08%                                             
          One or more SMs have a much higher number of active cycles than the average number of active cycles. Maximum  
          instance value is 33.03% above the average, while the minimum instance value is 15.39% below the average.     
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 22.11%                                             
          One or more SMSPs have a much higher number of active cycles than the average number of active cycles.        
          Maximum instance value is 33.12% above the average, while the minimum instance value is 15.31% below the      
          average.                                             
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 22.08%                                             
          One or more L1 Slices have a much higher number of active cyclesthan the average number of active cycles.    
          Maximum instance value is 33.03% above the average, while the minimum instance value is 15.39% below the      
          average.                                             
```
我其实使用NVIDIA Nsight Compute看的，有图形界面  
![1](image.png)  
看到这里的Duration是966.99us，计算下来，有效带宽大概是9.24GB/s，但是我查了一下，RTX5060 Laptop的峰值显存带宽能达到384GB/s，也就是我的利用率只有2.4%哈哈哈  
先看看有哪些地方可以提升的，我们现在看GPU有没有足够的并行工作。这部分就看block的数量，thread的数量和实际驻留的warp数量。这几个指标在报告中都可以看到。比如启动了多少个block，这是grid size，这里看到我的grid size是32；然后看block size，也就是每个block有多少个线程，我这里是128，感觉偏少了；然后是驻留的warp，这个要看Achieved Occupancy，他的意思是运行时实际平均驻留的warp占硬件上限的比例，我这里只有11.19%。  
我一开始以为的是，每个block的线程数不够多有影响，不过查询了一下发现我理解不对。因为原来的thread是128，有32个block。现在thread有256，block就只要16个；**然而一个block只能交给一个SM执行，而我的GPU有26个SM，32个block能让全部的SM分到任务，如果变成16个，反而有一些SM会空闲**。SM是流式多处理器，GPU把block分配给SM，SM再以warp为单位执行其中的线程。  
看来需要换一个思路了。我们要提升实际的warp驻留率，就需要给GPU提供更多可以同时执行的warp。  
**题目提示了可以用__shfl_down_sync。我需要理解一下什么是__shfl_down_sync**。查了一下，这个东西可以让一个线程直接取得同一warp内另一个线程的值，无需先写到共享内存。比如说每个线程都有自己的float sum，32个线程都执行了
```
float other = __shfl_down_sync(0xffffffffu, sum, 16);sum += other;
```
warp内，线程编号叫做lane，范围是0-31,这里的参数，0xffffffffu是32位全1，也就是整个warp的32个线程都参加，sum是每个线程提供自己的部分和，16是读取自身lane编号加16的线程的值。  
这个特性，有点像之前的合作求和。比如第一个加第129个，第二个加第130个，一直到第128个加第256个，现在就把256个的和分散到前面128个了；然后继续第一个加第65个，直到第64个加第128个；以此类推，最后就是都到第一个了。这个叫**树形归约**  
我们利用这个特点来对原来的算子进行修改。这里修改的原因就是，我一开始设计的是block中的一个thread负责计算一个元素，那么就需要一整行的数据，看我们之前的这个公式  
$$
Y[n] = \sum_{k=0}^{K-1}
X[k]\,
\left(q_{n,k} - Z[n,\lfloor k/G \rfloor]\right)\,
S[n,\lfloor k/G \rfloor]
$$

就是说一个Y[i]需要遍历一整行的X，还需要一些Q和Z。之前这个工作全是一个thread做的，就是他需要完成4096次计算。我们现在让几个thread一起合作来完成这个事情  
首先需要知道当前线程在warp中的编号。一个warp有32个thread，所以lane就是threadIdx.x % 32，而当前的warp在block中的编号warp_id就是threadIdx.x / 32了。另外，每个block中的warps应该是blockDim.x / 32。然后我们需要知道当前是哪一个warp  
所以用i表示，i就是blockIdx.x * warps_per_block + warp_id。  
在计算的时候，每个线程只计算这一行的一部分，比如lane0计算k = 32, 64,96,128这样，lane1就计算k = 1, 33, 65这样。然后后面再合并同一warp内的32个部分和。最后的完整的和存在lane0  
不过这样的话意味着每个warp计算一个输出，因此我们需要按照warp的数量来计算blocks   
新代码是  
```
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

    float sum = 0.0f;
    // 这里改成32个为stride来计算，后面合并
    for(int k = lane;k < K;k += 32) {
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
        // 顺便补一下之前忘记的转换
        float z = __half2float(zeros[group_index]);
        float s = __half2float(scales[group_index]);
        // 拿到现在的x
        float xv = __half2float(x[k]);
        
        float weight = (q - z) * s;
        sum += xv * weight;
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


```

新的测试结果是
```
[112779] python3.10@127.0.0.1
  gemv_kernel(const __half *, const unsigned char *, const __half *, const__half *, __half *, int, int, int) (1024, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.99
    SM Frequency                    Ghz         1.59
    Elapsed Cycles                cycle       284290
    Memory Throughput                 %        57.73
    DRAM Throughput                   %        14.29
    Duration                         us       177.82
    L1/TEX Cache Throughput           %        59.43
    L2 Cache Throughput               %         6.57
    SM Active Cycles              cycle    275148.92
    Compute (SM) Throughput           %        73.47
    ----------------------- ----------- ------------

    OPT   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the   
          compute pipelines are spending their time doing. Also, consider whether any computation is redundant and      
          could be reduced or moved to look-up tables.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                   1024
    Preferred Cluster Size                                         0
    Registers Per Thread             register/thread              40
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              26
    Stack Size                                                  1024
    Threads                                   thread          131072
    # TPCs                                                        13
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                3.28
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 25%                                             
          A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the    
          target GPU. The number of blocks in a wave depends on the numberof multiprocessors and the theoretical       
          occupancy of the kernel. This kernel launch results in 3 full waves and a partial wave of 89 thread blocks.   
          Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for  
          up to 25.0% of the total runtime of this kernel. Try launching agrid with no partial wave. The overall       
          impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware   
          Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for     
          more details on launch configurations.                                             

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block           12
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        87.62
    Achieved Active Warps Per SM           warp        42.06
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 12.38%                                             
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (87.6%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                             

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       279112
    Total DRAM Elapsed Cycles        cycle      7815168
    Average L1 Active Cycles         cycle    275148.92
    Total L1 Elapsed Cycles          cycle      7364570
    Average L2 Active Cycles         cycle    269373.31
    Total L2 Elapsed Cycles          cycle      4408176
    Average SM Active Cycles         cycle    275148.92
    Total SM Elapsed Cycles          cycle      7364570
    Average SMSP Active Cycles       cycle    274343.49
    Total SMSP Elapsed Cycles        cycle     29458280
    -------------------------- ----------- ------------
```
可以看到有一些提升了。我们的用时减少到了177.82us，计算下来，我们的有效带宽利用率达到13.1%  

第一步已经提升了，我们现在看看第二种方法。也就是提示中说的**探索 uint32_t, uint2, uint4 向量化加载（Vectorized Load）。**  
我先搜索了一下什么是**向量化加载**。他就是每个线程用一条更宽的加载指令，一次读取多个连续的数据元素。比如读取四个连续的32位整数，如果逐个读取就需要四条加载指令；但我们如果用CUDA的uint4来读取，就可以用一条128位的加载指令来读取。减少了加载指令的数量  
现在每轮读取一个uint8_t，里面有2个INT4，但是只用了里面的一个来计算。  
接下来我们保持一个warp计算一行，最后还是用shuffle求和。这里的第一轮，lane0读取0-3字节，计算0-7个权重；lane1读取4-7字节，计算8-15个权重。一个warp一轮就可以覆盖256个权重  
然后我们让一个thread负责计算好这些，合并到sum中。  
这么做就是减少了很多读取数据的时间，让warp中的线程合作读取  

这一次的算子代码是  
```
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
```
测试一下  

```
==PROF== Disconnected from process 159194
[159194] python3.10@127.0.0.1
  gemv_kernel(const __half *, const unsigned char *, const __half *, const__half *, __half *, int, int, int) (1024, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.99
    SM Frequency                    Ghz         1.55
    Elapsed Cycles                cycle       101977
    Memory Throughput                 %        87.88
    DRAM Throughput                   %        38.69
    Duration                         us        65.60
    L1/TEX Cache Throughput           %        92.78
    L2 Cache Throughput               %        17.25
    SM Active Cycles              cycle     96414.46
    Compute (SM) Throughput           %        58.88
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.  
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing L1 in the Memory Workload Analysis section.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                   1024
    Preferred Cluster Size                                         0
    Registers Per Thread             register/thread              36
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              26
    Stack Size                                                  1024
    Threads                                   thread          131072
    # TPCs                                                        13
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                3.28
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 25%                                             
          A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the    
          target GPU. The number of blocks in a wave depends on the numberof multiprocessors and the theoretical       
          occupancy of the kernel. This kernel launch results in 3 full waves and a partial wave of 89 thread blocks.   
          Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for  
          up to 25.0% of the total runtime of this kernel. Try launching agrid with no partial wave. The overall       
          impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware   
          Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for     
          more details on launch configurations.                                             

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block           12
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        87.24
    Achieved Active Warps Per SM           warp        41.87
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 12.76%                                             
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (87.2%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                             

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       278944
    Total DRAM Elapsed Cycles        cycle      2883584
    Average L1 Active Cycles         cycle     96414.46
    Total L1 Elapsed Cycles          cycle      2646830
    Average L2 Active Cycles         cycle     95744.19
    Total L2 Elapsed Cycles          cycle      1616800
    Average SM Active Cycles         cycle     96414.46
    Total SM Elapsed Cycles          cycle      2646830
    Average SMSP Active Cycles       cycle     95859.73
    Total SMSP Elapsed Cycles        cycle     10587320
    -------------------------- ----------- ------------
```
可以计算到我们这一次的有效带宽利用率达到了35.4%  
我忘了，既然都利用向量化加载了，怎么只加载了个w_q?我其实应该给x也来个向量化加载的，或许效率还能继续提升  
原本只是想跟w_q一样，然后x的读取改成float xv =  (packed_x >> (m * 16)) & 0xFFFF;不过发生了一些类型转换的错误，查询资料知道，这样会把提取出的16位当成整数，再转成float。比如half是1.0，编码是0x3C00,就会变成15360.0  
所以必须先按照half的位模式来解释，再转成float。也就是  
```
unsigned short bits = (packed_x >> (m * 16)) & 0xFFFFu;
float xv = __half2float(__ushort_as_half(bits));
```
经过这次的优化我们成功达到  
```
==PROF== Disconnected from process 7715
[7715] python3.10@127.0.0.1
  gemv_kernel(const __half *, const unsigned char *, const __half *, const__half *, __half *, int, int, int) (1024, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.99
    SM Frequency                    Ghz         1.53
    Elapsed Cycles                cycle        76516
    Memory Throughput                 %        50.74
    DRAM Throughput                   %        50.74
    Duration                         us        50.02
    L1/TEX Cache Throughput           %        31.19
    L2 Cache Throughput               %        22.73
    SM Active Cycles              cycle     71717.77
    Compute (SM) Throughput           %        75.60
    ----------------------- ----------- ------------

    OPT   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the   
          compute pipelines are spending their time doing. Also, consider whether any computation is redundant and      
          could be reduced or moved to look-up tables.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                   1024
    Preferred Cluster Size                                         0
    Registers Per Thread             register/thread              38
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              26
    Stack Size                                                  1024
    Threads                                   thread          131072
    # TPCs                                                        13
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                3.28
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 25%                                             
          A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the    
          target GPU. The number of blocks in a wave depends on the numberof multiprocessors and the theoretical       
          occupancy of the kernel. This kernel launch results in 3 full waves and a partial wave of 89 thread blocks.   
          Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for  
          up to 25.0% of the total runtime of this kernel. Try launching agrid with no partial wave. The overall       
          impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware   
          Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for     
          more details on launch configurations.                                             

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block           12
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        85.99
    Achieved Active Warps Per SM           warp        41.28
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 14.01%                                             
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (86.0%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                             

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       279032
    Total DRAM Elapsed Cycles        cycle      2199552
    Average L1 Active Cycles         cycle     71717.77
    Total L1 Elapsed Cycles          cycle      1985778
    Average L2 Active Cycles         cycle        71950
    Total L2 Elapsed Cycles          cycle      1227664
    Average SM Active Cycles         cycle     71717.77
    Total SM Elapsed Cycles          cycle      1985778
    Average SMSP Active Cycles       cycle     71343.67
    Total SMSP Elapsed Cycles        cycle      7943112
    -------------------------- ----------- ------------
```
经过计算，可以知道本地估算的有效带宽利用率**提升到了46%**  
Nvidia Nsight Compute给出的建议是**Compute is more heavily utilized than Memory: Look at the  Compute Workload Analysis section to see what the compute pipelines are spending their time doing. Also, consider whether any computation is redundant and could be reduced or moved to look-up tables**  
我们的Compute达到了75.6%而Memory只有50.74%，我们来看看能不能在计算方面进行一些优化  
**查了一下，ncu可以进行一些检查，需要运行以下指令**  
```
ncu   --target-processes all   --kernel-name regex:gemv_kernel   --launch-skip 4   --launch-count 1   --section SpeedOfLight \
--section ComputeWorkloadAnalysis \
--section InstructionStats \
--section SchedulerStats \
--section WarpStateStats  python test_local.py
```

得到  
```
==PROF== Disconnected from process 9816
[9816] python3.10@127.0.0.1
  gemv_kernel(const __half *, const unsigned char *, const __half *, const__half *, __half *, int, int, int) (1024, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.97
    SM Frequency                    Ghz         1.53
    Elapsed Cycles                cycle        76729
    Memory Throughput                 %        50.93
    DRAM Throughput                   %        50.93
    Duration                         us        49.95
    L1/TEX Cache Throughput           %        31.11
    L2 Cache Throughput               %        22.66
    SM Active Cycles              cycle     71916.62
    Compute (SM) Throughput           %        75.47
    ----------------------- ----------- ------------

    OPT   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the   
          compute pipelines are spending their time doing. Also, consider whether any computation is redundant and      
          could be reduced or moved to look-up tables.                                             

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         3.21
    Executed Ipc Elapsed  inst/cycle         3.02
    Issue Slots Busy               %        75.47
    Issued Ipc Active     inst/cycle         3.21
    SM Busy                        %        75.47
    -------------------- ----------- ------------

    INF   Shared FMA Heavy is the highest-utilized pipeline (58.3%) based on elapsed cycles in the workload, taking     
          into account the rates of its different instructions. It is a physical pipe and shared by the logical pipes   
          FMA Heavy and ALU Lite. It's dominated by its FMA Heavy sub-pipeline. It is well-utilized, but should not be  
          a bottleneck. Based on the number of executed instructions, the highest utilized pipeline (56.0%) is ALU      
          Heavy. It is part of the aggregated pipe ALU. Comparing the two,the overall pipeline utilization appears to  
          be caused by frequent, low-latency instructions. See the Profiling Guide                                      
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder) or hover over the          
          pipeline name to understand the workloads handled by each pipeline. The Instruction Statistics section shows  
          the mix of executed instructions for this workload.                                             

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        80.71
    Issued Warp Per Scheduler                        0.81
    No Eligible                            %        19.29
    Active Warps Per Scheduler          warp        10.36
    Eligible Warps Per Scheduler        warp         5.50
    ---------------------------- ----------- ------------

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        12.83
    Warp Cycles Per Executed Instruction           cycle        12.83
    Avg. Active Threads Per Warp                                31.87
    Avg. Not Predicated Off Threads Per Warp                    30.39
    ---------------------------------------- ----------- ------------

    WRN   The optional metric smsp__pcsamp_sample_count could not be found. Collecting it as an additional metric could 
          enable the rule to provide more guidance.                                             

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Local Memory Spilling Requests                  byte            0
    Shared Memory Spilling Requests                 byte            0
    Avg. Executed Instructions Per Scheduler        inst     57737.85
    Executed Instructions                           inst      6004736
    Avg. Issued Instructions Per Scheduler          inst     57737.85
    Issued Instructions                             inst      6004736
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 12.05%                                             
          This workload executes 524288 fused and 1069056 non-fused FP32 instructions. By converting pairs of non-fused 
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 34% (relative to its  
          current performance). 
```

也就是说，我们必须优化一下我们的计算过程。但是计算的公式他就是那么简单，还怎么优化呢？  
再看看我们的公式，我们的公式就是(q - z) * s，看看代码  
```
        // 这里的m代表当前包内的第m个INT4权重，范围是0-7，因为一个包就是8个INT4
        // 然后我们用按位与
        // 把4位提取出来，给q
        for(int m = 0; m < 8; m++) {
            int q = (packed >> (m * 4)) & 0xF;

            //必须先按照half的位模式来解释，再转成float
            unsigned short bits = (packed_x >> (m * 16)) & 0xFFFFu;
            float xv = __half2float(__ushort_as_half(bits));

            float weight = (q - z) * s;
            sum += xv * weight;
        }
```
**哦?突然发现这里面的z和s都是不变的，也就是每次循环的时候我们变得地方只有q是变化的，z和s都是不变的，是外面的。那么我们是不是可以优化掉这部分的计算呢？也就是我们可以把公式变成qs - zs，然后zs在外面算好，后面可以复用好几次。**  
**另外，Nvidia的文档可以发现，他有一个叫做FMA的东西，这东西好啊，它可以把权重的两部计算合并成一次，比如我们前面的q * s - z*s,z*s直接看成一个数字a，那相当于计算q*s-a**，本来是两步的，但是FMA可以合并成一次，接口是__fmaf_rn(a,b,c)，计算a\*b+c。  
那么前面所有的公式就可以变成FMA(q, s, -zs)  
那么我们的代码就可以改成  
```
        float z = __half2float(zeros[group_index]);
        float s = __half2float(scales[group_index]);

        // 在外面计算好，复用
        float zs = -z * s ;

        // 这里的m代表当前包内的第m个INT4权重，范围是0-7，因为一个包就是8个INT4
        // 然后我们用按位与
        // 把4位提取出来，给q
        for(int m = 0; m < 8; m++) {
            int q = (packed >> (m * 4)) & 0xF;

            //必须先按照half的位模式来解释，再转成float
            unsigned short bits = (packed_x >> (m * 16)) & 0xFFFFu;
            float xv = __half2float(__ushort_as_half(bits));

            float weight = __fmaf_rn(static_cast<float>(q), s, zs);
            sum += xv * weight;
        }
```

进行测试，有点失望，时间几乎没什么变化，只减少了3.4%，现在的利用率大概48.18%  
到这里我已经很难想到什么可以优化的地方了，而且这半天连50%都没到，也是很奇怪。只好去问AI了。AI给了我一个新的测试命令，也就是**在原来命令的基础上加上了--clock-control none**  
把测试命令换成  
```
ncu --target-processes all \
  --kernel-name regex:gemv_kernel \
  --launch-skip 4 --launch-count 1 --set basic \
  --clock-control none \
  python test_local.py --no-benchmark
```
再测试一下就会发现  
```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.97
    SM Frequency                    Ghz         2.61
    Elapsed Cycles                cycle        80440
    Memory Throughput                 %        82.70
    DRAM Throughput                   %        82.70
    Duration                         us        30.75
    L1/TEX Cache Throughput           %        32.18
    L2 Cache Throughput               %        23.96
    SM Active Cycles              cycle     69517.54
    Compute (SM) Throughput           %        66.39
    ----------------------- ----------- ------------
```
**这一次的Duration减少到了30.75us，U大概就是75.62%，**比上一次提升明显。可以看到的是**SM Frequency得到了大幅的提升，也就是说，GPU的SM频率提高了很多，我记得之前都是1.53左右的，这次到了2.61，很显然，就是SM频率提高实现的加速**  
