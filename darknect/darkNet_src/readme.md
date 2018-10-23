# 源程序解读
[源码分析参考 待深入](https://github.com/Ewenwan/darknet-1)

## 激活函数
```c
/*
**  根据输入的激活函数名称，返回标准（darknet定义的枚举类型）的激活函数类别
**  输入：s    C风格字符数组，激活函数名称，比如relu,logistic等等
**  返回：ACTIVATION   激活函数类别，枚举类型
**  说明：该函数仅仅通过匹配字符数组，返回标准的激活函数类别而已；
**       如果输入的激活函数名称未能识别，则统一使用RELU
*/
ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

/*
** 根据不同的激活函数类型，调用不同的激活函数处理单个输入元素x
** 输入： x    待处理的元素（单个）
**       a    激活函数类型
*/
float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

/**
** 用激活函数处理输入x中的每一个元素
** 输入： x    待处理的数组，一般为网络层每个神经元的加权输入Wx+b，在本函数中也相当于是输出（本地操作～）
**       n    x中含有多少个元素
**       a    激活函数类型
** 说明：该函数会逐个处理x中的元素，注意是逐个；该函数一般用于每一层网络的前向传播函数中，比如forward_connected_layer()等，
**      用在最后一步，该函数的输出即为每一层网络的输出
*/
void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    // 逐个处理x中的元素
    for(i = 0; i < n; ++i){
        // 根据不同的激活函数类型，调用不同的激活函数处理
        x[i] = activate(x[i], a);
    }
}

/*
** 根据不同的激活函数求取对输入的梯度（导数）
** 输入： x    激活函数接收的输入值
**       a    激活函数类型，包括的激活函数类型见activations.h中枚举类型ACTIVATION的定义
** 输出： 激活函数关于输入x的导数值
*/
float gradient(float x, ACTIVATION a)
{
    // 以下分别求取各种激活函数对输入的导数值，详见各个导数求取函数的内部注释
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

/*
** 计算激活函数对加权输入的导数，并乘以delta，得到当前层最终的delta（敏感度图）
** 输入： x    当前层的所有输出（维度为l.batch * l.out_c * l.out_w * l.out_h）
**       n    l.output的维度，即为l.batch * l.out_c * l.out_w * l.out_h（包含整个batch的）
**       ACTIVATION    激活函数类型
**       delta     当前层敏感度图（与当前成输出x维度一样）
** 说明1： 该函数不但计算了激活函数对于加权输入的导数，还将该导数乘以了之前完成大部分计算的敏感度图delta（对应元素相乘），
          因此调用改函数之后，将得到该层最终的敏感度图
** 说明2： 这里直接利用输出值求激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数，
          其关于输入的导数值都可以描述为输出值的函数表达式，
          比如对于Sigmoid激活函数（记作f(x)），其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x)，
          那么f(x)'=y*(1-y)，只需要输出值y就可以了，不需要输入x的值，
          （暂时不确定darknet中有没有使用特殊的激活函数，以致于必须要输入值才能够求出导数值，
          在activiation.c文件中，有几个激活函数暂时没看懂，也没在网上查到）。
** 说明3： 关于l.delta的初值，可能你有注意到在看某一类型网络层的时候，比如卷积层中的backward_convolutional_layer()函数，
          没有发现在此之前对l.delta赋初值的语句，
**        只是用calloc为其动态分配了内存，这样的l.delta其所有元素的值都为0,那么这里使用*=运算符得到的值将恒为0。
          是的，如果只看某一层，或者说某一类型的层，的确有这个疑惑，
**        但是整个网络是有很多层的，且有多种类型，一般来说，不会以卷积层为最后一层，而回以COST或者REGION为最后一层，
          这些层中，会对l.delta赋初值，又由于l.delta是由后
**        网前逐层传播的，因此，当反向运行到某一层时，l.delta的值将都不会为0.
*/
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 



```
