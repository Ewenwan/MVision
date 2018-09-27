// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.

// 修改：万有文
// opencv风格的数据结构的最小实现
// 大小结构体 Size 矩阵框结构体 Rect_ 交集 并集运算符重载 点结构体   Point_ 在头文件中实现
// 本文件实现一些 io 函数 和 Mat改变大小函数

#include "opencv.h"

#if NCNN_OPENCV

#include <stdio.h>

namespace cv {

// 读取图像=============================
Mat imread(const std::string& path, int flags)
{
    (void)flags;

    // read pgm/ppm
    // PPM->Portable PixMap 支持真彩色图形，可以读上面所有格式，输出PPM图形
    // PGM->Portable GreyMap 支持灰度图形，能够读PBM图形和PGM图形，输出PGM图形
    // PBM->Portable BitMap 支持单色图（1个像素位）
    // 由表头和图像数据两部分组成。表头数据各项之间用空格(空格键、制表键、回车键或换行键)隔开,表头由四部分组成:
    // ① 文件描述子: 指明文件的类型以及图像数据的存储方式;
    // ② 图像宽度;   
    // ③ 图像高度;
    // ④ 最大灰度值 或 颜色值.
    // 其中图像宽度、高度和最大值这三项是ASCII码十进制数.
/*
文件描述子 	类型 	  编码
P1 	        位图 	  ASCII
P2 	        灰度图  ASCII
P3 	        像素图  ASCII
P4 	        位图 	  二进制
P5 	        灰度图  二进制
P6 	        像素图  二进制
 
 */
    FILE* fp = fopen(path.c_str(), "rb");// 二进制 读
    if (!fp)
        return Mat();

    Mat m;

    char magic[3];// 文件头
    int w, h;
    int nscan = fscanf(fp, "%2s\n%d %d\n255\n", magic, &w, &h);
    if (nscan == 3 && magic[0] == 'P' && (magic[1] == '5' || magic[1] == '6'))
    {
        if (magic[1] == '5') // 灰度图
        {
            m.create(h, w, CV_8UC1);
        }
        else if (magic[1] == '6') // 像素图 彩色图
        {
            m.create(h, w, CV_8UC3);
        }
        if (m.empty())
        {
            fclose(fp);
            return Mat();
        }

        fread(m.data, 1, m.total(), fp); // 二进制 方式 读取数据内容
    }

    fclose(fp);

    return m;
}

// 保存图像 写图像============================
void imwrite(const std::string& path, const Mat& m)
{
    // write pgm/ppm
    FILE* fp = fopen(path.c_str(), "wb"); // 二进制写
    if (!fp)
        return;
// 写文件头====================
    if (m.channels() == 1)  // 单通道 灰度图
    {
        fprintf(fp, "P5\n%d %d\n255\n", m.cols, m.rows);
    }
    else if (m.channels() == 3)// 3通道 像素图 彩色图
    {
        fprintf(fp, "P6\n%d %d\n255\n", m.cols, m.rows);
    }
// 写图像数据==================
    fwrite(m.data, 1, m.total(), fp);// 二进制方式写数据

    fclose(fp);
}

#if NCNN_PIXEL
// 改变大小========================================================
void resize(const Mat& src, Mat& dst, const Size& size, float sw, float sh, int flags)
{
    int srcw = src.cols;// 源图像 宽高
    int srch = src.rows;

    int w = size.width;// 目标宽度和高度
    int h = size.height;

    if (w == 0 || h == 0)
    {
        w = srcw * sw; // 指定比例给出的 目标宽度和高度
        h = srch * sh;
    }

    if (w == 0 || h == 0)
        return;

    if (w == srcw && h == srch)// 和原图像尺寸 一致 不需要改变
    {
        dst = src.clone();
        return;
    }

    cv::Mat tmp(h, w, src.c);
    if (tmp.empty())
        return;
    // 双线性插值算法 改变图像大小
    if (src.c == 1)// 灰度图
        ncnn::resize_bilinear_c1(src.data, srcw, srch, tmp.data, w, h);
    else if (src.c == 3)// RGB/BGR 3通道彩色图
        ncnn::resize_bilinear_c3(src.data, srcw, srch, tmp.data, w, h);
    else if (src.c == 4)// RGBA/BGRA 4通道彩色图
        ncnn::resize_bilinear_c4(src.data, srcw, srch, tmp.data, w, h);

    dst = tmp;
}
#endif // NCNN_PIXEL

} // namespace cv

#endif // NCNN_OPENCV
