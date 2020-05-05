# Tengine 高性能神经网络推理引擎

[源码](https://github.com/Ewenwan/Tengine)

[Tengine 推断引擎：树莓派也能玩转深度学习](https://shumeipai.nxez.com/2018/12/07/tengine-inference-engine-raspberry-pi-deep-learning.html)

[Tengine Winograd快速卷积算法 ](https://github.com/Ewenwan/Winograd_tutorial_python)

[基于ARM-v8的Tengine GEMM 矩阵乘法 汇编优化 教程 ](https://github.com/Ewenwan/Tengine_gemm_tutorial)

[Tengine 白皮书](https://cdn-file.aijishu.com/494/739/494739128-5d51139b186ca.pdf?_upt=c49f6b9e1588562426)

# 编译

>  安装相关工具

    sudo apt-get instal git cmake

    git 是一个版本控制系统，稍后将用来从 github 网站上下载Tengine的源码
    cmake 是一个编译工具，用来产生make过程中所需要的Makefile文件
    
> 安装支持库

sudo apt-get install libprotobuf-dev protobuf-compiler libboost-all-dev libgoogle-glog-dev libopencv-dev libopenblas-dev

    protobuf 是一种轻便高效的数据存储格式，这是caffe各种配置文件所使用的数据格式
    boost 是一个c++的扩展程序库，稍后Tengine的编译依赖于该库
    google-glog 是一个google提供的日志系统的程序库
    opencv 是一个开源的计算机视觉库
    openblas 是一个开源的基础线性代数子程序库

> 特点

重点加速卷积等最为耗时的算子 convolution/FC/Pooling 支持多种卷积计算模式 GEMM/Direct/Winogrid

手工汇编调优，CPU微架构极致优化，Dataflow多线程加速，适配ARM A7/A17/A35/A53/A72/A73/A55/A76

支持F32/F16/Int8动态量化混合精度计算模式


## 框架

老接口：

1. 初始化 init_tengine();

2. 载入模型，创建图 create_graph(nullptr, "tengine", tm_file)  普通设备

       create_graph(nullptr, "tiny", tm_mem)     // mcu  stm32
       create_graph(nullptr, "zhouyi", tm_file)   // 周易 AIPU
       create_graph(nullptr, "nnie", tm_file, config)     // 海思 nnie 3519  3516
       create_graph(nullptr, "rk3399pro", tm_mem)   // rk3399pro  AIPU

3. 设置图属性 和 输入数据
     
       get_graph_input_tensor(graph, 0, 0);
       set_graph_attr(graph, "low_mem_mode", &val, sizeof(val));
       
4. 预推理 
       
       prerun_graph(graph)
       
5. 正式运行
       
       run_graph(graph, 1)
       
6. 清理

       release_graph_tensor(input_tensor);
       release_graph_tensor(output_tensor);
       postrun_graph(graph);
       destroy_graph(graph);

       release_tengine();
       
新接口（类似ncnn的）：



## **gemm  矩阵乘法（全连接层、卷积核和输入展开后的矩阵乘法、卷积winogrid变换后的矩阵乘法）**

矩阵乘法的加速运算 A[M K] * B[K N]  ======  C[M N]

纯c实现:
```C

void gemm_pure_c(float* A, float* B, float* C,int m,int n,int k)
{
   for(int i=0;i<M;i++) // 安装目标 C 矩阵维度遍历 先行维M  
    {
       for(int j=0; j<N;j++) // 再 列维 N    主要是 矩阵 是 列优先排布
       {
           C[i*n+j]=0.f;
           for(int p=0; p<K;p++) //A 矩阵每一行K个元素
           {
                C[i*N + j] += A[i*K + p] * B[p*N + j];
                // C的i行j列   A的i行p列    B的p行j列
           }
       }
    }
}

```

openblas 函数实现

数据并行SIMD  NEON 向量优化

手写向量汇编优化

## **winogrid变换卷积运算**


      输入矩阵转换
               ----> 元素乘法 gemm算法  ----> 输出转换
      权重矩阵转换
      
      
1. define transform matrix
   ```python
   # kernel转换
   G_F23 = np.array([
        [ 1.0,  0.0, 0.0 ],
        [ 0.5,  0.5, 0.5 ],
        [ 0.5, -0.5, 0.5 ],
        [ 0.0,  0.0, 1.0 ]])
    # 输入转换矩阵
    Bt_F23 = np.array([
        [ 1.0,  0.0, -1.0,  0.0 ],
        [ 0.0,  1.0,  1.0,  0.0 ],
        [ 0.0, -1.0,  1.0,  0.0 ],
        [ 0.0,  1.0,  0.0, -1.0 ]])
    # 输出转换矩阵    
    At_F23 = np.array([
        [ 1.0, 1.0,  1.0,  0.0 ],
        [ 0.0, 1.0, -1.0, -1.0 ]])
   ```
2. compute transformation for input, kernel, output
   ```python
    # 输入矩阵转换   g' = G*g*G转置 
    def trans_kernel(g):
        return np.dot(np.dot(G_F23,g),G_F23.T)
    # 权重kernel转换 d' = B转置*d*B转
    def trans_input(d):
        return np.dot(np.dot(Bt_F23,d),Bt_F23.T)
    # o' = g' * d'
    # 输出转换 o = A转置*o'*A转
    def trans_output(r):
        return np.dot(np.dot(At_F23,r),At_F23.T)
   ```
3. do conv_winof23, conv_direct
   ```python
    def wino_f23(kernel,input):
        tran_inp = trans_input(input)
        tran_ker = trans_kernel(kernel)
        mid = tran_inp * tran_ker
        out = trans_output(mid)
        return out

    def conv_direct(kernel,input):
        out=np.zeros((2,2))
        for h in range(2):
            for w in range(2):
                out[h,w]=np.sum(input[h:h+3,w:w+3]*kernel)
        return out
   ```
      




