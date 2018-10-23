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
## 均值 方差 归一化 差平方 softmax归一化
```C
/*
** 有组织的计算输入数据x的平均值，输出的mean是一个矢量，比如如果x是多张3通道的图片，那么mean的维度就为通道数3
** （也即每张输入图片会得到3张特征图）,为方便，我们称这三个通道分别为第一，第二，第三通道，由于每次训练输入的都是一个batch的图片，
** 因此最终会输出batch张三通道的图片，mean中的第一个元素就是第一个通道上全部batch张输出特征图所有元素的平均值，依次类推
** 本函数的主要用处之一应该就是实现batch normalization的第一步了！
** 输入： 
**       x         包含所有数据，比如l.output，其包含的元素个数为l.batch*l.outputs
**       batch     一个batch中包含的图片张数，即l.batch
**       filters   该层神经网络的滤波器个数，也即该层网络输出图片的通道数（比如对卷积网络来说，就是核的个数了）
**       spatial   该层神经网络每张输出特征图的尺寸，也即等于l.out_w*l.out_h
**       mean      求得的平均值，维度为filters，也即每个滤波器对应有一个均值（每个滤波器会处理所有图片）
** 说明： 该函数的具体调用可以参考：batchnorm_layer.c中的forward_batchnorm_layer()函数
** 说明2：mean_cpu()函数是一个纯粹的数学计算函数，有组织的计算x中某些数据的均值，x的具体存储结构视具体情况而定，
**       在写注释时，主要参考了batchnorm_layer.c中的forward_batchnorm_layer()函数对该函数的调用，
**       因此有些注释就加上了一些具体含义，结合这些含义会有助于理解，但还是要记住，这是一个一般的数学计算函数，
**       不同地方调用该函数可能有不同的含义。
** 说明3：均值是哪些数据的均值？x中包含了众多数据，mean中的每个元素究竟对应x中哪些数据的平均值呢？
**       此处还是结合batchnorm_layer.c中的forward_batchnorm_layer()函数的调用来解释，
**       其中的x为l.output，有l.batch行，每行有l.out_c*l.out_w*l.out_h个元素，每一行又可以分成
**       l.out_c行，l.out_w*l.out_h列，那么l.mean中的每一个元素，是某一个通道上所有batch的输出的平均值
**       （比如卷积层，有3个核，那么输出通道有3个，每张输入图片都会输出3张特征图，可以理解每张输出图片是3通道的，
**       若每次输入batch=64张图片，那么将会输出64张3通道的图片，而mean中的每个元素就是某个通道上所有64张图片
**       所有元素的平均值，比如第1个通道上，所有64张图片像素平均值）
** 说明4：在全连接层的前向传播函数中：sptial=1，因为全连接层的输出可以看作是1*1的特征图
*/
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    // scale即求均值中的分母项
    float scale = 1./(batch * spatial);
    int i,j,k;
    // 外层循环次数为filters，也即mean的维度，每次循环将得到一个平均值
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        // 中层循环次数为batch，也即叠加每张输入图片对应的某一通道上的输出
        for(j = 0; j < batch; ++j){
            // 内层循环即叠加一张输出特征图的所有像素值
            for(k = 0; k < spatial; ++k){
                // 如果理解了上面的注释，下面的偏移是很显然的
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];// 和
            }
        }
        // 除以该均值所涉及元素的总个数，得到平均值
        mean[i] *= scale;
    }
}

/*
** 计算输入x中每个元素的方差（大致的过程和上面的mean_cpu类似，不再赘述）
** 本函数的主要用处之一应该就是batch normalization的第二步了！
** 输入： 
**       x         包含所有数据，比如l.output，其包含的元素个数为l.batch*l.outputs
**       batch     一个batch中包含的图片张数，即l.batch
**       filters   该层神经网络的滤波器个数，也即该层网络输出图片的通道数（比如对卷积网络来说，就是核的个数了）
**       spatial   该层神经网络每张输出特征图的尺寸，也即等于l.out_w*l.out_h
**       mean      求得的平均值，维度为filters，也即每个滤波器对应有一个均值（每个滤波器会处理所有图片）
*/
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    // 为什么计算方差分母要减去1呢？参考这里吧：https://www.zhihu.com/question/20983193
    // 事实上，在统计学中，往往采用的方差计算公式都会让分母减1,这时因为所有数据的方差是基于均值这个固定点来计算的，
    // 对于有n个数据的样本，在均值固定的情况下，其采样自由度为n-1（只要n-1个数据固定，第n个可以由均值推出）
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                // 每个元素减去均值求平方
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}
// 归一化
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);// 去均值除方差归一化
            }
        }
    }
}

/*
** 计算预测数组与真实标签数组中每对元素的l2范数值，或者说是计算squared error，
** 注意此函数，并没有求和，没有将所有误差加起来，而是对网络输出的每个元素计算误差的平方值
** 输入：n       输出元素个数，也即pred中的元素个数，也是truth中的元素个数
**      pred    网络最终的输出值，或者说网络的预测值，其中输出元素个数为n（也即最后一层网络神经元个数为n）
**      truth   真实标签值，其中元素个数为n（也即最后一层网络神经元个数为n）
**      delta   相当于本函数的输出，为网络的敏感度图（一般为cost_layer.c的敏感度图）
**      error   相当于本函数的输出，包含每个输出元素的squared error
** 说明：这个函数一般被cost_layer.c调用，用于计算cost_layer每个输出的均方误差，除此之外，还有一个重要的操作，
**      就是计算网络最后一层的敏感度图，在darknet中，最后一层比较多的情况是cost_l
*/
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i]; // 差
        error[i] = diff * diff;          // 差平方
        delta[i] = diff;
    }
}


/*
** 输入： input   一组输入图片数据（含义见下面softmax_cpu()注释，下同）
**       n       一组输入数据中含有的元素个数n=l.inputs/l.groups
**       temp    温度参数，关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
**       stride  跨度
**       output  这一组输入图片数据对应的输出（也即l.output中与这一组输入对应的某一部分）
** 说明：本函数实现的就是标准的softmax函数处理，唯一有点变化的就是在做指数运算之前，将每个输入元素减去了该组输入元素中的最大值，以增加数值稳定性，
**      关于此，可以参考博客：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/，
**      这篇博客写的不错，博客中还提到了softmax-loss，此处没有实现（此处实现的也即博客中提到的softmax函数，将softmax-loss分开实现了）。
*/
void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    // 赋初始最大值为float中的最小值-FLT_MAX（定义在float.h中）
    float largest = -FLT_MAX;
    // 寻找输入中的最大值，至于为什么要找出最大值，是为了数值计算上的稳定，详细请戳：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
    // 这篇博客写的不错，博客在接近尾声的时候，提到了为什么要减去输入中的最大值。
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        // 在进行指数运算之间，如上面博客所说，首先减去最大值（当然温度参数也要除）
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;                       // 求和
        output[i*stride] = e;           // 并将每一个输入的结果保存在相应的输出中
    }
    // 最后一步：归一化转换为概率（就是softmax函数的原型～），最后的输出结果保存在output中
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

/**
 * @brief 对输入input进行softmax处理得到输出output
 * @param input    softmax层所有输入数据（包含整个batch的），即net.input（上一层的输出）
 * @param n        一组输入数据中含有的元素个数n=l.inputs/l.groups
 * @param batch    一个batch中所含有的图片张数（等于net.batch）
 * @param batch_offset    一张输入图片含有的元素个数，即值等于l.inputs（所以叫做batch_offset，目的是要借助该参数在input中整张整张照片移位）
 * @param groups   一张输入图片的元素被分成了几组，值为l.groups（这个参数由配置文件指定，如果未指定，则默认为1）,这个参数暂时还没遇到怎么用，
 *                 大部分的网络值都为1,也即相当于没有这个参数
 * @param group_offset    值等于n，组偏移（在每张输入图片元素中整组整组偏移）
 * @param stride  跨度，这个参数类似于axpy_cpu()函数中的INCX参数，一定注意不同于卷积层中的l.stride，这个参数是指按照stride间隔从每组输入
 *                数据中抽取元素，即会抽取所有索引为stride倍数的输入元素，而其他的输入元素，实际没有用到；stride=1时，显然，相当于没有这个参数，
 *                所有输入数据都用到了（这个参数在softmax_layer层中，相当于没用，因为在forward_softmax_layer()中，调用该函数时，stride已经
 *                被写死为1,并不能改，不知道还有没有其他地方使用了这个参数）
 * @param temp     softmax的温度参数l.temperature，关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
 * @param output   经softmax处理之后得到的输出l.output（即概率），与input具有相同的元素个数（见make_softmax_layer()），其实由此也可知，
 *                stride的值必然为1,不然output的元素个数肯定少于input的元素个数（所以对于softmax来说，感觉设置stride是没有必要的，有点自相矛盾的意思）
 * @note 以上注释针对的是softmax_layer，另有不同地方调用本函数的在调用处进行详细注释；上面的注释出现了新的量词单位，这里厘清一下关系：输入input
 *        中包括batch中所有图片的输入数据，其中一张图片具有inputs个元素，一张图片的元素又分成了groups组，每组元素个数为n=l.inputs/l.groups
*/
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    // 遍历batch中的每张图片
    for(b = 0; b < batch; ++b){
        // 每张图片又按组遍历：一组一组遍历
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}
```


