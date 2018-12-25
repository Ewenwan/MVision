# PyTorch
[参考](https://blog.csdn.net/zzulp/article/details/80573331)

    Pytorch是torch的python版本，
    是由Facebook开源的神经网络框架。
    与Tensorflow的静态计算图不同，
    pytorch的计算图是动态的，
    可以根据计算需要实时改变计算图。
    
    安装：
    
    pip install torch torchvision  # for python2.7
    pip3 install torch torchvision  # for python3
    
## 特点

    1. Numpy风格的Tensor操作。pytorch中tensor提供的API参考了Numpy的设计，
         因此熟悉Numpy的用户基本上可以无缝理解，并创建和操作tensor，
         同时torch中的数组和Numpy数组对象可以无缝的对接。
    2. 变量自动求导。在一序列计算过程形成的计算图中，
         参与的变量可以方便的计算自己对目标函数的梯度。
         这样就可以方便的实现神经网络的后向传播过程。
    3. 神经网络层与损失函数优化等高层封装。
         网络层的封装存在于torch.nn模块，
         损失函数由torch.nn.functional模块提供，
         优化函数由torch.optim模块提供。

## Tensor类型
    Torch 定义了七种 CPU tensor 类型和八种 GPU tensor 类型:

    Data type                    CPU tensor         GPU tensor
    32-bit floating point    torch.FloatTensor   torch.cuda.FloatTensor
    64-bit floating point    torch.DoubleTensor  torch.cuda.DoubleTensor
    16-bit floating point    torch.HalfTensor    torch.cuda.HalfTensor
    8-bit integer (unsigned) torch.ByteTensor    torch.cuda.ByteTensor
    8-bit integer (signed)   torch.CharTensor    torch.cuda.CharTensor
    16-bit integer (signed)  torch.ShortTensor   torch.cuda.ShortTensor
    32-bit integer (signed)  torch.IntTensor     torch.cuda.IntTensor
    64-bit integer (signed)  torch.LongTensor    torch.cuda.LongTensor
 ### 创建接口
    方法名                            说明
    Tensor()                  直接从参数构造一个的张量，参数支持list,numpy数组
    eye(row, column)          创建指定行数，列数的二维单位tensor
    linspace(start,end,count) 在区间[s,e]上创建c个tensor
    logspace(s,e,c)           在区间[10^s, 10^e]上创建c个tensor
    ones(*size)               返回指定shape的张量，元素初始为1
    zeros(*size)              返回指定shape的张量，元素初始为0
    ones_like(t)              返回与t的shape相同的张量，且元素初始为1
    zeros_like(t)             返回与t的shape相同的张量，且元素初始为0
    arange(s,e,sep)           在区间[s,e)上以间隔sep生成一个序列张量
    
###  随机采样
    方法名                  说明
    rand(*size)         在区间[0,1)返回一个均匀分布的随机数张量
    uniform(s,e)        在指定区间[s,e]上生成一个均匀分布的张量
    randn(*size)        返回正态分布N(0,1)取样的随机数张量
    normal(means, std   返回一个正态分布N(means, std)
 
    
### 数学操作
    这些方法均为逐元素处理方法

    方法名                   说明
    abs                     绝对值
    add                     加法
    addcdiv(t, v, t1, t2)   t1与t2的按元素除后，乘v加t
    addcmul(t, v, t1, t2)   t1与t2的按元素乘后，乘v加t
    ceil                    向上取整，天花板
    floor                   向下取整，地面
    clamp(t, min, max)      将张量元素限制在指定区间
    exp                     指数
    log                     对数
    pow                     幂
    mul                     逐元素乘法
    neg                     取反
    sigmoid                 指数归一化   exp(-xi)/sum(exp(-xi))
    sign                    取符号
    sqrt                    开根号
    tanh	
 
