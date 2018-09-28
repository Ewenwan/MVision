// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
// 批规范化=====
// 去均值 归一化 缩放 平移 合在一起=============
// 各个通道均值 mean_data = sum(xi)/m;
// 各个通道方差 var_data  = sum((xi - mean_data)^2)/m;
// xi‘ = ( xi - mean_data )/(sqrt(var_data + eps));  // 去均值，除以方差，归一化
// yi = slope_data * xi'  + bias_data; //  缩放 + 平移
// 写成一起=====================
// yi = slope_data / (sqrt(var_data + eps)) * xi  + bias_data - slope_data*mean_data/(sqrt(var_data + eps));
// b = slope_data / (sqrt(var_data + eps)) = slope_data /sqrt_var;
// a = bias_data - slope_data*mean_data/(sqrt(var_data + eps)) = bias_data - slope_data*mean_data/sqrt_var;
// yi = b * xi + a;

#include "batchnorm.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm)

BatchNorm::BatchNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int BatchNorm::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.f); // 一个极小值，放置除0

    return 0;
}

int BatchNorm::load_model(const ModelBin& mb)
{
  // 缩放系数=====
    slope_data = mb.load(channels, 1);
    if (slope_data.empty())
        return -100;

    // 均值数据====
    mean_data = mb.load(channels, 1);
    if (mean_data.empty())
        return -100;
    
   // 方差数据=====
    var_data = mb.load(channels, 1);
    if (var_data.empty())
        return -100;// out of memory
   
   // 平移系数======
    bias_data = mb.load(channels, 1);
    if (bias_data.empty())
        return -100;

    a_data.create(channels);
    if (a_data.empty())
        return -100;
    b_data.create(channels);
    if (b_data.empty())
        return -100;

    // 去均值 归一化 合在一起=============
    // 各个通道均值 mean_data = sum(xi)/m
    // 各个通道方差 var_data     = sum((xi - mean_data)^2)/m
    // xi‘ = ( xi - mean_data )/(sqrt(var_data + eps))  // 去均值，除以方差，归一化
    // yi = slope_data * xi'  + bias_data  //  缩放 + 平移
    // 写成一起=====================
    // yi = slope_data / (sqrt(var_data + eps)) * xi  + bias_data - slope_data*mean_data/(sqrt(var_data + eps)) 
    // b = slope_data / (sqrt(var_data + eps)) = slope_data /sqrt_var;
    // a = bias_data - slope_data*mean_data/(sqrt(var_data + eps)) = bias_data - slope_data*mean_data/sqrt_var;
    // yi = b * xi + a
    for (int i=0; i<channels; i++)
    {
        float sqrt_var = sqrt(var_data[i] + eps);// 各个通道 标准差
        a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
        b_data[i] = slope_data[i] / sqrt_var;
    }

    return 0;
}

// 前向传播函数=====可直接在输入blob上修改=======
int BatchNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;// 1维数据，w为数据总数量============

        float* ptr = bottom_top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];// 归一化 + 缩放 + 平移 一起======
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<h; i++)
        {
            float* ptr = bottom_top_blob.row(i);// 每一行数据===============
            float a = a_data[i];
            float b = b_data[i];

            for (int j=0; j<w; j++)
            {
                ptr[j] = b * ptr[j] + a;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

            for (int i=0; i<size; i++)
            {
                ptr[i] = b * ptr[i] + a;
            }
        }
    }

    return 0;
}

} // namespace ncnn
