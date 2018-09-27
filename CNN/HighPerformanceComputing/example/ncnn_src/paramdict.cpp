// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
// 修改： 万有文
// 层参数解析===============
// 读取二进制格式、字符串格式、密文格式的参数文件

/*
参数字典，每一层的意义不一样：
      数据输入层 Input            data             0 1 data 0=227 1=227 2=3   图像宽度×图像高度×通道数量
      卷积层    Convolution  ...   0=64     1=3      2=1    3=2     4=0    5=1    6=1728           
               0输出通道数 num_output() ; 1卷积核尺寸 kernel_size();  2空洞卷积参数 dilation(); 3卷积步长 stride(); 
               4卷积填充pad_size();             5卷积偏置有无bias_term();  6卷积核参数数量 weight_blob.data_size()；
                                                               C_OUT * C_in * W_h * W_w = 64*3*3*3 = 1728
      池化层    Pooling      0=0       1=3       2=2        3=0       4=0
                          0池化方式:最大值、均值、随机     1池化核大小 kernel_size();     2池化核步长 stride(); 
                          3池化核填充 pad();   4是否为全局池化 global_pooling();
      激活层    ReLU       0=0.000000     下限阈值 negative_slope();
               ReLU6      0=0.000000     1=6.000000 上下限
      
      综合示例：
      0=1 1=2.5 -23303=2,2.0,3.0
      
      数组关键字 : -23300 
      -(-23303) - 23300 = 3 表示该参数在参数数组中的index
      后面的第一个参数表示数组元素数量，2表示包含两个元素
*/

#include <ctype.h>
#include "paramdict.h"
#include "platform.h"

namespace ncnn {

ParamDict::ParamDict()
{
    use_winograd_convolution = 1;
    use_sgemm_convolution = 1;
    use_int8_inference = 1;

    clear();
}
 // 获取参数==========================
int ParamDict::get(int id, int def) const
{
    return params[id].loaded ? params[id].i : def;
}

float ParamDict::get(int id, float def) const
{
    return params[id].loaded ? params[id].f : def;
}

Mat ParamDict::get(int id, const Mat& def) const
{
    return params[id].loaded ? params[id].v : def;
}

// 设置参数=========================
void ParamDict::set(int id, int i)
{
    params[id].loaded = 1;
    params[id].i = i;
}

void ParamDict::set(int id, float f)
{
    params[id].loaded = 1;
    params[id].f = f;
}

void ParamDict::set(int id, const Mat& v)
{
    params[id].loaded = 1;
    params[id].v = v;
}

// 清空参数========================
void ParamDict::clear()
{
    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
    {
        params[i].loaded = 0;
        params[i].v = Mat();
    }
}

#if NCNN_STDIO
#if NCNN_STRING

// 判断字符串是否为小数 
static bool vstr_is_float(const char vstr[16])
{
    // look ahead for determine isfloat
    for (int j=0; j<16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
}

int ParamDict::load_param(FILE* fp)
{
    clear();

//     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    int id = 0;
    while (fscanf(fp, "%d=", &id) == 1)// 读取 等号前面的 key=========
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;// 数组 关键字 -23300  得到该参数在参数数组中的 index
        }
        
// 是以 -23300 开头表示的数组===========
        if (is_array)
        {
            int len = 0;
            int nscan = fscanf(fp, "%d", &len);// 后面的第一个参数表示数组元素数量，5表示包含两个元素
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read array length fail\n");
                return -1;
            }

            params[id].v.create(len);

            for (int j = 0; j < len; j++)
            {
                char vstr[16];
                nscan = fscanf(fp, ",%15[^,\n ]", vstr);//按格式解析字符串============
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict read array element fail\n");
                    return -1;
                }

                bool is_float = vstr_is_float(vstr);// 检查该字段是否为 浮点数的字符串

                if (is_float)
                {
                    float* ptr = params[id].v;
                    nscan = sscanf(vstr, "%f", &ptr[j]);// 转换成浮点数后存入参数字典中
                }
                else
                {
                    int* ptr = params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);// 转换成 整数后 存入字典中
                }
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict parse array element fail\n");
                    return -1;
                }
            }
        }
// 普通关键字=========================
        else
        {
            char vstr[16];
            int nscan = fscanf(fp, "%15s", vstr);// 获取等号后面的 字符串
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read value fail\n");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);// 判断是否为浮点数

            if (is_float)
                nscan = sscanf(vstr, "%f", &params[id].f); // 读入为浮点数
            else
                nscan = sscanf(vstr, "%d", &params[id].i);// 读入为整数
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict parse value fail\n");
                return -1;
            }
        }

        params[id].loaded = 1;// 设置该 参数以及载入
    }

    return 0;
}
#endif // NCNN_STRING

// 读取 二进制格式的 参数文件===================
int ParamDict::load_param_bin(FILE* fp)
{
    clear();

//     binary 0
//     binary 100
//     binary 1
//     binary 1.250000
//     binary 3 | array_bit
//     binary 5
//     binary 0.1
//     binary 0.2
//     binary 0.4
//     binary 0.8
//     binary 1.0
//     binary -233(EOP)

    int id = 0;
    fread(&id, sizeof(int), 1, fp);// 读入一个整数长度的 index

    while (id != -233)// 结尾
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;// 数组关键字对应的 index
        }
// 是数组数据=======
        if (is_array)
        {
            int len = 0;
            fread(&len, sizeof(int), 1, fp);// 数组元素数量

            params[id].v.create(len);

            float* ptr = params[id].v;
            fread(ptr, sizeof(float), len, fp);// 按浮点数长度*数组长度 读取每一个数组元素====
        }
// 是普通数据=======
        else
        {
            fread(&params[id].f, sizeof(float), 1, fp);// 按浮点数长度读取 该普通字段对应的元素
        }

        params[id].loaded = 1;

        fread(&id, sizeof(int), 1, fp);// 读取 下一个 index
    }

    return 0;
}
#endif // NCNN_STDIO

// 读取密文格式的参数文件===========================
int ParamDict::load_param(const unsigned char*& mem)
{
    clear();

    int id = *(int*)(mem);
    mem += 4;

    while (id != -233)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = *(int*)(mem);
            mem += 4;

            params[id].v.create(len);

            memcpy(params[id].v.data, mem, len * 4);
            mem += len * 4;
        }
        else
        {
            params[id].f = *(float*)(mem);
            mem += 4;
        }

        params[id].loaded = 1;

        id = *(int*)(mem);
        mem += 4;
    }

    return 0;
}

} // namespace ncnn
