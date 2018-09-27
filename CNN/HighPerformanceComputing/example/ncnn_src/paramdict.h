// Tencent is pleased to support the open source community by making ncnn available.
//
// 修改： 万有文
// 层参数解析===============
// 读取二进制格式、字符串格式、密文格式的参数文件==

#ifndef NCNN_PARAMDICT_H
#define NCNN_PARAMDICT_H

#include <stdio.h>
#include "mat.h"
#include "platform.h"

// at most 20 parameters
#define NCNN_MAX_PARAM_COUNT 20

namespace ncnn {

class Net;
class ParamDict
{
public:
    // empty
    ParamDict();
    
    // 获取参数==========================
    // get int
    int get(int id, int def) const;
    // get float
    float get(int id, float def) const;
    // get array
    Mat get(int id, const Mat& def) const;
    
     // 设置参数=========================
    // set int
    void set(int id, int i);
    // set float
    void set(int id, float f);
    // set array
    void set(int id, const Mat& v);

public:
    int use_winograd_convolution;
    int use_sgemm_convolution;
    int use_int8_inference;

protected:
    friend class Net;// 友元类 Net类 可以调用 ParamDict类
    
    // 清空参数========================
    void clear();

#if NCNN_STDIO
#if NCNN_STRING
    int load_param(FILE* fp);// 读取字符串格式参数文件 
#endif // NCNN_STRING
    int load_param_bin(FILE* fp);// 读取二进制格式参数文件
#endif // NCNN_STDIO
    int load_param(const unsigned char*& mem);// 读取密文格式的参数文件

protected:
    struct
    {
        int loaded;// 解析完成标志
        union { int i; float f; };// 整形 / 浮点型
        Mat v;// 数组
    } params[NCNN_MAX_PARAM_COUNT];// 参数数组
};

} // namespace ncnn

#endif // NCNN_PARAMDICT_H
