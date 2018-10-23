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

## 卷积 矩阵乘法
```c
/*
*  将输入图像im的channel通道上的第row行，col列像素灰度值加上val（直接修改im的值，因此im相当于是返回值）
** 输入： im         输入图像
**       channels   输入图像的im通道数（这个参数没用。。。）
**       height     输入图像im的高度（行）
**       width      输入图像im的宽度（列）
**       row        需要加上val的像素所在的行数（补零之后的行数，因此需要先减去pad才能得到真正在im中的行数）
**       col        需要加上val的像素所在的列数（补零之后的列数，因此需要先减去pad才能得到真正在im中的列数）
**       channel    需要加上val的像素所在的通道数
**       pad        四周补0长度
**       val        像素灰度添加值
*/
void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    // 边界检查：超过边界则不作为返回
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}

//This one might be too, can't remember.
/*
** 此函数与im2col_cpu()函数的流程相反，目地是将im2col_cpu()函数重排得到的图片data_col恢复至正常的图像矩阵排列，
   并与data_im相加，最终data_im相当于是输出值，
** 要注意的是，data_im的尺寸是在函数外确定的，且并没有显示的将data_col转为一个与data_im尺寸相同的矩阵，
  （data_im初始所有元素值都为0）。
** 得到的data_im尺寸为l.c*l.h*l.w，即为当前层的输入图像尺寸，上一层的输出图像尺寸，按行存储，
   可视为l.c行，l.h*l.w列，即其中每行对应一张输出特征图的敏感度图（实际上这还不是最终的敏感度，
** 还差一个环节：乘以激活函数对加权输入的导数，这将在下一次调用backward_convolutional_laye时完成）。
**
** 举个例子：第L-1层每张输入图片（本例子只分析单张输入图片）的输出为5*5*3（3为输出通道数），
           第L层共有2个卷积核，每个卷积核的尺寸为3*3，stride = 2,
**         第L-1层的输出是第L层的输入，第L层的2个卷积核同时对上一层3个通道的输出做卷积运算，为了做到这一点，需要调用im2col_cpu()函数将
**         上一层的输出，也就是本层的输入重排为27行4列的图，也就是由5*5*3变换至27*4，你会发现总的元素个数变多了（75增多到了98），
**         这是因为卷积核stride=2,小于卷积核的尺寸3,因此卷积在两个连续位置做卷积，会有重叠部分，而im2col_cpu()函数为了便于卷积运算，
**         完全将其铺排开来，并没有在空间上避免重复元素，因此像素元素会增多。
**         此外，之所以是27行，是因为卷积核尺寸为3*3，而上一层的输出即本层输入有3个通道，为了同时给3个通道做卷积运算，
**         需要将3个通道上的输入一起考虑，即得到3*3*3行，4列是因为对于对于5*5的图像，使用3*3的卷积核，stride=2的卷积跨度，
**         最终会得到2*2的特征图，也就是4个元素。除了调用im2col_cpu()对输入图像做重排，
**         相应的，也要将所有卷积核重排成一个2*27的矩阵，为什么是2呢？ 因为有两个卷积核，为了做到同时将两个卷积核作用到输入图像上，
**         需要将两个核合到一个矩阵中，每个核对应一行，因此有2行，那为什么是27呢？每个核 元素个数不是3*3=9吗？是的，
**         但是考虑到要同时作用到3个通道上，所以实际一个卷积核有9*3=27个元素。综述，得到2*27的卷积核矩阵与27*4的输入图像矩阵，
**         两个矩阵相乘，即可完成将2个卷积核同时作用于3通道的输入图像上（非常方便，不枉前面非这么大劲的重排！），
           最终得到2*4的矩阵，这2*4矩阵又代表这什么呢？2代表这有两个输出图（对应2个卷积核，即l.out_c=2），
**         每个输出图占一行，4代表这每个输出图元素有4个（前面说了，每个卷积核会得到2*2的特征图，即l.out_h=l.out_w=2）。
           这个例子说到这，只说完了前向传播部分，可以看出im2col_cpu()这个函数的重要性。
**         而此处的col2im_cpu()是一个逆过程，主要用于反向传播中，由L层的敏感度图(sensitivity map，
**         可能每个地方叫的不一样，此处参考博客：https://www.zybuluo.com/hanbingtao/note/485480)反向求得第L-1层的敏感度图。
           顺承上面的例子，第L-1层的输出是一个5*5*3（l.w=l.h=5,l.c=3）的矩阵，也就是敏感度图的维度为5*5*3
**         （每个输出元素，对应一个敏感度值），第L层的输出是一个2*4的矩阵，敏感度图的维度为2*4，假设已经计算得到了
**         第L层的2*4的敏感度图，那么现在的问题是，如何由第L层的2*4敏感度图以及2个卷积核（2*27）反向获取第L-1层的敏感度图呢？
           上面给的博客链接给出了一种很好的求解方式，但darknet并不是这样做的，为什么？因为前面有im2col_cpu()，
**         im2col_cpu()函数中的重排方式，使得我们不再需要博客中提到的将sensitivity map还原为步长为1的sensitivity map，
**         只需再使用col2im_cpu()就可以了！过程是怎样的呢，看backward_convolutional_layer()函数中if(net.delta)中的语句就知道了，
           此处仅讨论col2im_cpu()的过程，在backward_convolutional_layer()已经得到了data_col，
**         这个矩阵含有了所有的第L-1层敏感度的信息，但遗憾的是，不能直接用，需要整理，因为此时的data_col还是一个
**         27*4的矩阵，而我们知道第L-1层的敏感度图是一个5*5*3的矩阵，如何将一个27*4的矩阵变换至一个5*5*3的矩阵是本函数要完成的工作，
           前面说到27*4元素个数多于5*5*3, 很显然要从27*4变换至5*5*3，肯定会将某些元素相加合并
**        （下面col2im_add_pixel()函数就是干这个的），具体怎样，先不说，先来看看输入参数都代表什么意思吧：

** 输入： data_col    backward_convolutional_layer()中计算得到的包含上一层所有敏感度信息的矩阵，
                      行数为l.n*l.size*l.size（l代表本层/当前层），列数为l.out_h*l.out_w（对于本例子，
                      行数为27,列数为4,上一层为第L-1层，本层是第L层） 
**       channels    当前层输入图像的通道数（对于本例子，为3）
**       height      当前层输入图像的行数（对于本例子，为5）
**       width       当前层输入图像的列数（对于本例子，为5）
**       ksize       当前层卷积核尺寸（对于本例子，为3）
**       stride      当前层卷积跨度（对于本例子，为2）
**       pad         当前层对输入图像做卷积时四周补0的长度
**       data_im     经col2im_cpu()重排恢复之后得到的输出矩阵，也即上一层的敏感度图，
                     尺寸为l.c * l.h * l.w（刚好就是上一层的输出当前层输入的尺寸，对于本例子，5行5列3通道），
                     注意data_im的尺寸，是在本函数之外就已经确定的，不是在本函数内部计算出来的，
                     这与im2col_cpu()不同，im2col_cpu()计算得到的data_col的尺寸都是在函数内部计算得到的，
                     并不是事先指定的。也就是说，col2im_cpu()函数完成的是指定尺寸的输入矩阵往指定尺寸的输出矩阵的转换。
** 原理：原理比较复杂，很难用文字叙述，博客：https://www.zybuluo.com/hanbingtao/note/485480
        中基本原理说得很详细了，但是此处的实现与博客中并不一样，所以具体实现的原理此处简要叙述一下，具体见个人博客。
        第L-1层得到l.h*l.w*l.c输出，也是第L层的输入，经L层卷积及激活函数处理之后，
        得到l.out_h*l.out_w*l.out_c的输出，也就是由l.h*l.w*l.c-->l.out_h*l.out_w*l.out_c，
        由于第L层有多个卷积核，所以第L-1层中的一个输出元素会流入到第L层多个输出中，除此之外，
        由于卷积核之间的重叠，也导致部分元素流入到第L层的多个输出中，这两种情况，都导致第L-1层中的某个敏感度会与第L层多个输出有关，
        为清晰，还是用上面的例子来解释，第L-1层得到5*5*3(3*25)的输出，第L层得到2*2*2（2*4）的输出，
        在backward_convolutional_layer()已经计算得到的data_col实际是27*2矩阵与2*4矩阵相乘的结果，
        为方便，我们记27*2的矩阵为a，记2*4矩阵为b，那么a中一行（2个元素）与b中一列（2个元素）相乘对应这什么呢？
        对应第一情况，因为有两个卷积核，使得L-1中一个输出至少与L层中两个输出有关系，经此矩阵相乘，得到27*4的矩阵，
        已经考虑了第一种情况（27*4这个矩阵中的每一个元素都是两个卷积核影响结果的求和），
        那么接下来的就是要考虑第二种情况：卷积核重叠导致的一对多关系，具体做法就是将data_col中对应相同像素的值相加，
        这是由im2col_cpu()函数决定的（可以配合im2col_cpu()来理解），因为im2col_cpu()将这些重叠元素也铺陈保存在data_col中，
        所以接下来，只要按照im2col_cpu()逆向将这些重叠元素的影响叠加就可以了，
       大致就是这个思路，具体的实现细节可能得见个人博客了（这段写的有点罗嗦～）。     
*/
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im) 
{
    int c,h,w;
    // 当前层输出图的尺寸（对于上面的例子，height_col=2,width_col=2）
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    // 当前层每个卷积核在所有输入图像通道上的总元素个数（对于上面的例子，channels_col=3*3*3=27）
    // 注意channels_col实际是data_col的行数
    int channels_col = channels * ksize * ksize;

    // 开始遍历：外循环遍历data_col的每一行（对于上面的例子，data_col共27行）
    for (c = 0; c < channels_col; ++c) {

        // 列偏移，卷积核是一个二维矩阵，并按行存储在一维数组中，利用求余运算获取对应在卷积核中的列数，比如对于
        // 3*3的卷积核，当c=0时，显然在第一列，当c=5时，显然在第2列，当c=9时，在第二通道上的卷积核的第一列
        int w_offset = c % ksize;

        // 行偏移，卷积核是一个二维的矩阵，且是按行（卷积核所有行并成一行）存储在一维数组中的，
        // 比如对于3*3的卷积核，处理3通道的图像，那么一个卷积核具有27个元素，每9个元素对应一个通道上的卷积核（互为一样），
        // 每当c为3的倍数，就意味着卷积核换了一行，h_offset取值为0,1,2
        int h_offset = (c / ksize) % ksize;

        // 通道偏移，channels_col是多通道的卷积核并在一起的，比如对于3通道，3*3卷积核，每过9个元素就要换一通道数，
        // 当c=0~8时，c_im=0;c=9~17时，c_im=1;c=18~26时，c_im=2
        // c_im是data_im的通道数（即上一层输出当前层输入的通道数），对于上面的例子，c_im取值为0,1,2
        int c_im = c / ksize / ksize;

        // 中循环与内循环和起来刚好遍历data_col的每一行（对于上面的例子，data_col的列数为4,height_col*width_col=4）
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {

                // 获取在输出data_im中的行数im_row与列数im_col
                // 由上面可知，对于3*3的卷积核，h_offset取值为0,1,2,当h_offset=0时，会提取出所有与卷积核第一行元素进行运算的像素，
                // 依次类推；加上h*stride是对卷积核进行行移位操作，比如卷积核从图像(0,0)位置开始做卷积，那么最先开始涉及(0,0)~(3,3)
                // 之间的像素值，若stride=2，那么卷积核进行行移位一次时，下一行的卷积操作是从元素(2,0)（2为图像行号，0为列号）开始
                int im_row = h_offset + h * stride;
                // 对于3*3的卷积核，w_offset取值也为0,1,2，当w_offset取1时，会提取出所有与卷积核中第2列元素进行运算的像素，
                // 实际在做卷积操作时，卷积核对图像逐行扫描做卷积，加上w*stride就是为了做列移位，
                // 比如前一次卷积其实像素元素为(0,0)，若stride=2,那么下次卷积元素起始像素位置为(0,2)（0为行号，2为列号）
                int im_col = w_offset + w * stride;

                // 计算在输出data_im中的索引号
                // 对于上面的例子，im_row的取值范围为0~4,im_col从0~4，c从0~2（其中h_offset从0~2,w_offset从0~2, h从0~1,w从0~1）
                // 输出的data_im的尺寸为l.c * l.h * lw，对于上面的例子，为3*5*5,因此，im_row,im_col,c的取值范围刚好填满data_im

                // 获取data_col中索引为col_index的元素，对于上面的例子，data_col为27*4行，按行存储
                // col_index = c * height_col * width_col + h * width_col + w逐行读取data_col中的每一个元素。
                // 相同的im_row,im_col与c_im可能会对应多个不同的col_index，这就是卷积核重叠带来的影响，处理的方式是将这些val都加起来，
                // 存在data_im的第im_row - pad行，第im_col - pad列（c_im通道上）中。
                // 比如上面的例子，上面的例子，如果固定im_row = 0, im_col =2, c_im =0，由c_im = 0可以知道c在0~8之间，
                // 由im_row=0,可以确定h = 0, h_offset =0，
                // 可以得到两组：1)w_offset = 0, w = 1; 2) w_offset = 2, w =0，第一组，则可以完全定下：c=0,h=0,w=1，
                // 此时col_index=1，由第二组，可完全定下：c=2,h=0,w=0，
                // 此时col_index = 2*2*2=8
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];

                // 从data_im找出c_im通道上第im_row - pad行im_col - pad列处的像素，使其加上val
                // height, width, channels都是上一层输出即当前层输入图像的尺寸，也是data_im的尺寸（对于本例子，三者的值分别为5,5,3）,
                // im_row - pad,im_col - pad,c_im都是某一具体元素在data_im中的行数与列数与通道数
                // （因为im_row与im_col是根据卷积过程计算的，
                // 所以im_col和im_row中实际还包含了补零长度pad，需要减去之后，才是原本的没有补零矩阵data_im中的行列号）
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}


```

## 双向链表的实现
```c
// 节点=====
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;
// 链表====
typedef struct list{
    int size;
    node *front;
    node *back;
} list;

list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

/*
**	将val指针插入list结构体l中，这里相当于是用C实现了C++中的list的元素插入功能
**	流程：	list中并不直接含有void*类型指针，但list中含有的node指针含有void*类型指针，
**		  因此，将首先创建一个node指针new，而后将val传给new，最后再把new插入list指针l中
**	说明： void*指针可以接收不同数据类型的指针，因此第二个参数具体是什么类型的指针得视情况而定
**	调用： 该函数在众多地方调用，很多从文件中读取信息存入到list变量中，都会调用此函数，
**		  注意此函数类似C++的insert()插入方式；而在option_list.h中的opion_insert()函数，
**		  有点类似C++ map数据结构中的按值插入方式，比如map[key]=value，两个函数操作对象都是list变量，
**		  只是操作方式略有不同。
*/
void list_insert(list *l, void *val)
{
	// 定义一个node指针并动态分配内存
	node *new = malloc(sizeof(node));
	// 将输入的val指针赋值给new中的val元素，注意，这是指针复制，共享地址，二者都是void*类型指针
	new->val = val;
	new->next = 0;

	// 下面的链表嵌套主要注意一下
	// 如果l的back元素为空指针，说明l到目前为止，还没有存入数据（只是在函数外动态分配了内存，并没有赋有效值），
	// 这样，令l的front为new（此后front将不会再变，除非删除），显然此时new的前面没有node，因此new->prev=0
	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		// 如果已经插入了第一个元素，那么往下嵌套，注意这里操作的是指针，互有影响
		// 新插入的node赋给back的下一个node next，
		// 同时对于新的node new来说，其前一个node为l->back
		// 一定注意要相互更新（链表上的数据位置关系都是相对的）
		l->back->next = new;
		new->prev = l->back;
	}
	// 更新back的值
	// 不管前面进行了什么操作，每次插入new，都必须更新l的back为当前new，因为链表结构，
	// 新插入的元素永远在最后，此处back就是链表中的最后一个元素，front是链表中的第一个元素，
	// 链表中的第一个元素在第一次插入元素之后将不会再变（除非删除）
	l->back = new;
	// l中存储的元素个数加1
	++l->size;
}

/*
**	释放节点内存，注意节点node是结构体指针，其中含有自引用，需要递归释放内存
**	输入：	n	需要释放的node指针，其内存是动态分配的
**	注意：	不管什么程序，堆内存的释放都是一个重要的问题，darknet中的内存释放，值得学习！
**		  node结构体含有自引用，输入的虽然是一个节点，但实则是一个包含众多节点的链表结构，
**		  释放内存时，一定要递归释放，且格外注意内存释放的顺序
*/
void free_node(node *n)
{
	// 输入的node n是链表结构中的第一个（最前面的）node，因此是从前到后释放内存的，
	// 在释放每个node的之前，必须首先获取该node的下一个node，否则一旦过早释放，
	// 该节点之后的node将无从访问，无法顺利释放，这就可能造成内存泄漏
	node *next;
	// 遍历链表上所有的node，依次释放，直至n为空指针，说明整个链表上的节点释放完毕
	while(n) {
		// 释放n之前，首先获取n的下一个节点的指针，存至next中
		next = n->next;

		// 释放当前node内存
		free(n);
		// 将next赋值给n，成为下一个释放的节点
		n = next;
	}
}

/*
**	释放链表list变量的内存，主要调用free_node()函数以及C中的free()函数
**	输入：	l	要释放内存的list指针，其内存是动态分配的（堆内存）
**	注意：	list是个结构体，其中又嵌套有node结构体指针，因此需要递归释放内存。
**	      像这样嵌套动态分配内存的指针数据（结构体，还有C++中的类等），
**		  一定要作用到最底层，不能只在表层释放
*/
void free_list(list *l)
{
	// list中含有两种数据类型，其中主要一种是node，由于node存在自引用，组成了一个链表结构，
	// 因此需要调用free_node()函数递归释放内存
	free_node(l->front);
	// 释放完node内存之后，在释放l本身中含有的其他数据类型，即int类型
	// 同样，node的释放必须在释放l之前，否则将访问不到node，无法顺利释放node的内存，
	// 很可能造成内存泄漏
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

/*
**	将list变量l中的每个node的val提取出来，存至一个二维字符数组中，返回
**	输入：	l	list类型变量，此处的list类型变量包含众多节点，每个节点的值就是我们需要提取出来的信息
**	返回：返回类型为void**指针，因此，返回后，需要视具体情况进行指针类型墙转，比如转为char**（二维字符数组）
**  调用：data.c->get_labels()调用，目的是为了从包含数据集所有物体类别名称信息的l中提取所有物体名称，存入二维字符数组中并返回（这样做是为了便于访问）
*/
void **list_to_array(list *l)
{
	// a是一个二维字符数组：第一维可以看作是行，第二维可以看作是列，每一行相当于是一个字符数组（类似C++中的string）
    // a是一个二维字符数组，每个a[i]是一个一维字符数组，不过l中node的值是void*类型的，所以分配内存时，直接获取void*的大小就可以了，
	// calloc()是C语言函数，第一个参数是元素的数目，第二个则是每个元素的大小（字节）
	void **a = calloc(l->size, sizeof(void*));
    int count = 0;

	// 获取l中的首个节点
    node *n = l->front;
	// 遍历l中所有的节点，将各节点的值抓取出来，放到a中
    while(n){
		// 为每一个字符数组赋值：直接用指针赋值（n->val是指针，a[count++]也是指针，且都是void*型），不再需要为每一个a[i]动态分配内存，这里极好地使用了指针带来的便利
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

```

