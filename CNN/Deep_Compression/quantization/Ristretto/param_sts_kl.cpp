/*
a minimal sample case (CHW) 通道*高度*宽度
输入 input = 1x1x16, 16 random value          1通道*1高度*16宽度   随机值
卷积1 conv1 weight = 8x1x1x1, random weight   8输出*1输入*1*1卷积  随机值
卷积2 conv2 weight = 8x1x1x1, random weight   8输出*1输入*1*1卷积  随机值
输出 output = 8x1x16  8通道 * 1高度 * 16宽度

网络：
input -> conv1 -> output1
input -> conv2 -> output2
even the input data is the same for conv1 and conv2
the input scale factor may be different for minimize the output1/output2 KL divergence

测试：
编译： g++ -O3 -fopenmp -march=native dis.cpp -o dis
运行： ./dis 
input absmax = 1.973593
weight absmax = 1.665431
output absmax = 3.286882
min_i_scale = 7.58274307e+01
min_w_scale = 1.05593918e+02
min_KL_div = 328.901031
-------------------------
input absmax = 1.973593
weight absmax = 1.883400
output absmax = 3.717065
min_i_scale = 9.58458710e+01
min_w_scale = 1.07303162e+02
min_KL_div = 347.122253
-------------------------

*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <algorithm>

const int ZBIN = 2048;
const int DB = 512; // 直方图 bins 数量=============

// 产生 -2~2随机数
// always return -2 ~ 2
static inline float random_float()
{
    return ((float) rand()) / (float) RAND_MAX * 4 - 2; // 0~4 -2 ----> -2~2
}

// KL散度(Kullback-Leibler_divergence)
// 俗称KL距离，常用来衡量两个概率分布的距离。
// 参考 https://blog.csdn.net/sallyyoung_sh/article/details/54406615
/// 1. 根据shannon的信息论，给定一个字符集的概率分布，我们可以设计一种编码，
//   使得表示该字符集组成的字符串平均需要的比特数最少。
//   假设这个字符集是X，对x∈X，其出现概率为P(x)， 
//   那么其最优编码平均需要的比特数等于这个字符集的 信息熵entropy：
//  H(X) = -SUM( p(X=x) * log(p(X=x)) )
// a.当log以2为底的时候称之为 bits,结果可以视为多少个二进制位可以表示该变量
// b.当log以e为底的时侯称之为 nats

/// 2. KL divergence (KL距离)
//这个值是用来衡量两个分布之间相异度的，
// 具体来说，假设有k个状态的两个离散分布p,q，则:
// KL(p||q) = sum(pk * log(pk/qk)) = sum(pk * log(pk)) - sum(pk * log(qk))
//  = -H(p) + H(p,q)
// 其中H(p,q)称为 交叉熵 (cross entropy) = -sum(pk * log(qk))
// 交叉熵可以看作是当我们用模型q来编码来自模型p的变量时所需的平均bits(如果log以2为底的话)
// 所以，有H(p)=H(p,p),所以KL距离就可以看做是：
// 用模型q来编码来自模型p的变量所需的额外bits！
// 因为是“额外的”，所以 KL的距离的值一定大于0，KL=0当且仅当p=q
static inline float KL_divergence(const float* a, const float* b, int size)
{
    float KL = 0.f;
    for (int i=0; i<size; i++)
    {
        KL += a[i] * log(a[i] / b[i]);// 用 模型b 来编码来自 模型 a 的变量所需的额外bits！
    }
    return KL;
}

// 浮点数直接取整到 -128~127===========
static inline signed char float2int8(float v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (signed char)round(v);
}

// 数据直方图分布信息==================
static void distribution(const float* data, int size, float* hist, int bins, float absmax)
{
	// 初始化每个直方图的值=============
	
    for (int i=0; i<bins; i++)
    {
        hist[i] = 0.000001;// eps
    }
	
    // 遍历每一个数据放入直方图中==========
    for (int i=0; i<size; i++)
    {
        int b = (data[i] + absmax) / (absmax * 2) * bins;//计算 数据落入直方图中的bin id
		// b = (d[i] / (2*absmax) + 0.5) * bins
		// 总数据区域 为 -absmax 到 absmax，宽度为 2*absmax，对应直方图bins 的 id 为 0~bins-1
		// +0.5是为了平移 起点

        if (b < 0) b = 0;
        if (b >= bins) b = bins-1;
        hist[b] += 1;// 落入哪一个，哪一个就+1
    }
}

// 获取数据中的最大值======================
static inline float get_absmax(const float* data, int size)
{
    // find data abs max
    float absmax = 0.f;
    for (int i=0; i<size; i++)
    {
        absmax = std::max(absmax, fabs(data[i]));// 数据中的最大值
    }
    return absmax;
}

static float quantize(const float* data, int size, const float* w, int wsize, const float* output)
{
    float absmax = get_absmax(data, size);// 输入数据最大值
    float wabsmax = get_absmax(w, wsize);// 卷积参数最大值
    float outabsmax = get_absmax(output, size*wsize);// 输出数据最大值

    fprintf(stderr, "input absmax = %f\n", absmax);
    fprintf(stderr, "weight absmax = %f\n", wabsmax);
    fprintf(stderr, "output absmax = %f\n", outabsmax);

    // hist for output  统计输出数据的直方图分布信息==============
    float* hist = new float[DB];// 512
    distribution(output, size*wsize, hist, DB, outabsmax);

    //
    float min_KL_div = FLT_MAX;
    float min_i_scale = 0.f;
    float min_w_scale = 0.f;

    float* qout = new float[size*wsize];// 量化的输出数据
    float* qhist = new float[DB];// 量化后的数据的 直方图分布

    signed char* qdata = new signed char[size]; // 输入量化成     -128~127
    signed char* qw = new signed char[wsize];   // 卷积参数量化成 -128~127

    for (int z=0; z<ZBIN; z++)// 2048
    {
//         fprintf(stderr, "z = %d\n", z);

// 计算输入参数的不同量化尺度，相当于不断调整量化范围==============
        float i_scale = 127 / (absmax * (ZBIN-z)/ZBIN);// 迭代不同的量化尺度
		// 量化尺度 逐渐增加，相当于不断缩小 量化上限值 127/abamax --->127*2048/abamax (相当于量化范围缩减到 -1/2048*absmax ~ 1/2048*absmax)
		// 127为 一半的 量化值，所以是 absmax * 
		// (ZBIN-z)/ZBIN 最大值的比例 1 ...(2048-i)/2048.. 0
		
// 对输入参数进行量化=======================
        for (int i=0; i<size; i++)
        {
            qdata[i] = float2int8(data[i] * i_scale);//乘以缩放比例后，取整
        }

// 对卷积参数 按 不同的量化尺度进行量化=======================
        for (int y=0; y<ZBIN; y++)
        {
            float w_scale = 127 / (wabsmax * (ZBIN-y)/ZBIN);// 卷积参数的不同量化尺度========
            
			// 对 卷积参数 进行量化============
            for (int i=0; i<wsize; i++)
            {
                qw[i] = float2int8(w[i] * w_scale);//乘以缩放比例后，取整 
            }

            float rescale = 1 / (i_scale * w_scale);// 反量化缩放系数======

            #pragma omp parallel for
            for (int i=0; i<size; i++)
            {
                float* qoutptr = qout + i*wsize;// 输出参数的 指针
                for (int j=0; j<wsize; j++)
                {
                    qoutptr[j] = qdata[i] * qw[j] * rescale;// 矩阵乘法, 整数乘法后，再乘以反量化缩放系数，得到原有浮点数结果
                }
            }

            // build distribution
			// 统计量化后的 输出数据的 直方图统计分布===========
            distribution(qout, size*wsize, qhist, DB, outabsmax);
            
			// 计算 未使用量化参数得到的 结构 和 使用量化参数 计算得到的结果 计算KL散度距离
            float KL_div = KL_divergence(qhist, hist, DB);
            if (KL_div < min_KL_div)
            {
                min_KL_div = KL_div;   // 记录最小散度对应的 散度值
                min_i_scale = i_scale; // 输入参数 的 量化尺度
                min_w_scale = w_scale; // 卷积参数 的 量化尺度
            }
        }
    }

	// 清理内存
    delete[] qdata;
    delete[] qw;

    delete[] qout;
    delete[] qhist;

    delete[] hist;

    fprintf(stderr, "min_i_scale = %.8e\n", min_i_scale);
    fprintf(stderr, "min_w_scale = %.8e\n", min_w_scale);
    fprintf(stderr, "min_KL_div = %f\n", min_KL_div);
    fprintf(stderr, "-------------------------\n");

    return min_i_scale;// 返回输入数据的量化 尺度
}

int main()
{
    srand(time(NULL));// 随机数种子

    const int size = 16; //  输入 数据维度
    const int wsize = 8;

    // generate random input data
    float* data = new float[size];
    for (int i=0; i<size; i++)
    {
        data[i] = random_float();// 为每个输入数据 产生随机值
    }

    // generate two weight
    float* w1 = new float[wsize];
    float* w2 = new float[wsize];
    for (int i=0; i<wsize; i++)
    {
        w1[i] = random_float();// 为每个 卷积参数 产生随机值
        w2[i] = random_float();
    }

    // calculate output  计算两种 矩阵乘法 的 浮点数结果 
    float* output1 = new float[size*wsize];
    float* output2 = new float[size*wsize];
    for (int i=0; i<size; i++)
    {
        output1[i] = 0.f;
        output2[i] = 0.f;
        for (int j=0; j<wsize; j++)
        {
            output1[i*wsize+j] += data[i] * w1[j]; // 矩阵乘法，对应元素乘积
            output2[i*wsize+j] += data[i] * w2[j];
        }
    }
	

    // quantize input for minimum output KL divergence
    float i_scale1 = quantize(data, size, w1, wsize, output1);// 计算 KL散度最低的 量化尺度
    float i_scale2 = quantize(data, size, w2, wsize, output2);

    delete[] data;
    delete[] output1;
    delete[] output2;

    return 0;
}
