// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//   ncnn框架接口：
//       注册 用户定义的新层 Net::register_custom_layer();
//       网络载入 模型参数   Net::load_param();
//       载入     模型权重   Net::load_model();
//			 网络blob 输入       Net::input();
//       网络前向传          Net::forward_layer();    被Extractor::extract() 执行
//			 创建网络模型提取器   Net::create_extractor();
//       模型提取器提取某一层输出 Extractor::extract();

#ifndef NCNN_NET_H
#define NCNN_NET_H

#include <stdio.h>
#include <vector>// 容器
#include "blob.h" // 层的输入输出blob定义
#include "layer.h"// 层定义
#include "mat.h"  // 矩阵库
#include "platform.h"// 平台定义

namespace ncnn {

class Extractor;
class Net
{
public:
    // empty init
    Net();
    // clear and destroy
    ~Net();

#if NCNN_STRING
    // register custom layer by layer type name
    // return 0 if success
    int register_custom_layer(const char* type, layer_creator_func creator);
#endif // NCNN_STRING
    // register custom layer by layer type
    // return 0 if success
    int register_custom_layer(int index, layer_creator_func creator);

#if NCNN_STDIO
#if NCNN_STRING
    // load network structure from plain param file
    // return 0 if success
    int load_param(FILE* fp);
    int load_param(const char* protopath);
#endif // NCNN_STRING
    // load network structure from binary param file
    // return 0 if success
    int load_param_bin(FILE* fp);
    int load_param_bin(const char* protopath);

    // load network weight data from model file
    // return 0 if success
    int load_model(FILE* fp);
    int load_model(const char* modelpath);
#endif // NCNN_STDIO

    // load network structure from external memory
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_param(const unsigned char* mem);

    // reference network weight data from external memory
    // weight data is not copied but referenced
    // so external memory should be retained when used
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_model(const unsigned char* mem);

    // unload network structure and weight data
    void clear();

    // construct an Extractor from network
    Extractor create_extractor() const;

public:
    // enable winograd convolution optimization
    // improve convolution 3x3 stride1 performace, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    int use_winograd_convolution;// 快速卷积 Coppersmith-Winograd算法，时间复杂度是O(n^2.38)

    // enable sgemm convolution optimization
    // improve convolution 1x1 stride1 performace, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    int use_sgemm_convolution;

    // enable quantized int8 inference
    // use low-precision int8 path for quantized model
    // changes should be applied before loading network structure and weight
    // enabled by default
    int use_int8_inference;

protected:
    friend class Extractor;// 友元类
#if NCNN_STRING
    int find_blob_index_by_name(const char* name) const;  //
    int find_layer_index_by_name(const char* name) const; // 
    int custom_layer_to_index(const char* type);                   // 
    Layer* create_custom_layer(const char* type);                // 
#endif // NCNN_STRING
    Layer* create_custom_layer(int index);                              // 
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, Option& opt) const;// 

protected:
    std::vector<Blob> blobs;      // 
    std::vector<Layer*> layers;  // 

    std::vector<layer_registry_entry> custom_layer_registry; //
};

class Extractor
{
public:
    // enable light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    void set_light_mode(bool enable);

    // set thread count for this extractor
    // this will overwrite the global setting
    // default count is system depended
    void set_num_threads(int num_threads);

    // set blob memory allocator
    void set_blob_allocator(Allocator* allocator);

    // set workspace memory allocator
    void set_workspace_allocator(Allocator* allocator);

#if NCNN_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const Mat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, Mat& feat);
#endif // NCNN_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const Mat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, Mat& feat);

protected:
    friend Extractor Net::create_extractor() const;
    Extractor(const Net* net, int blob_count);

private:
    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;
};

} // namespace ncnn

#endif // NCNN_NET_H
