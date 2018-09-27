// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
// 修改：万有文
// opencv风格的数据结构的最小实现
//    大小结构体 Size 
//    矩阵框结构体 Rect_ 交集 并集运算符重载
//		点结构体     Point_
//	  矩阵结构体   Mat     深拷贝 浅拷贝 获取指定矩形框中的roi 读取图像 写图像 双线性插值算法改变大小

#ifndef NCNN_OPENCV_H
#define NCNN_OPENCV_H

#include "platform.h"

#if NCNN_OPENCV

#include <algorithm>
#include <string>
#include "mat.h"

// minimal opencv style data structure implementation
namespace cv
{
  
// 大小结构体Size，默认为public=======
struct Size
{
    Size() : width(0), height(0) {}// 初始化为空
    Size(int _w, int _h) : width(_w), height(_h) {}// 参数初始化
    int width;  // 图像宽度
    int height; // 图像高度
};

// 矩阵框结构体 Rect_ 默认为public=======
template<typename _Tp>
struct Rect_
{
    Rect_() : x(0), y(0), width(0), height(0) {}// 初始化为空
    Rect_(_Tp _x, _Tp _y, _Tp _w, _Tp _h) : x(_x), y(_y), width(_w), height(_h) {}// 参数初始化

    _Tp x;// 左上角点 (x,y)
    _Tp y;
    _Tp width; // 矩形框宽度和高度
    _Tp height;

    // area       面积函数
    _Tp area() const
    {
        return width * height;
    }
};

// 矩形框结构体的 &=  交集= 运算符重载=============================
// a &= b ; a = a&b; 把a框和b框的交集框给到a
template<typename _Tp> 
static inline Rect_<_Tp>& operator &= ( Rect_<_Tp>& a, const Rect_<_Tp>& b )
{
    _Tp x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);    // 交集的左上角点
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;    // 交集的宽 和 高
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1; a.y = y1;
    if( a.width <= 0 || a.height <= 0 )
        a = Rect_<_Tp>();
    return a;
}

// 矩形框结构体的 |=  并集= 运算符重载=============================
template<typename _Tp> 
static inline Rect_<_Tp>& operator |= ( Rect_<_Tp>& a, const Rect_<_Tp>& b )
{
    _Tp x1 = std::min(a.x, b.x), y1 = std::min(a.y, b.y);       // 并集 的 左上角点
    a.width = std::max(a.x + a.width, b.x + b.width) - x1;  // 并集 的 宽 和 高
    a.height = std::max(a.y + a.height, b.y + b.height) - y1;
    a.x = x1; a.y = y1;
    return a;
}

// 求交集的 运算符重载，返回一个新的矩形框===========================
template<typename _Tp> 
static inline Rect_<_Tp> operator & (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a; // 定义一个新的矩阵框
    return c &= b;
}

// 求并集的 运算符重载，返回一个新的矩形框
template<typename _Tp> 
static inline Rect_<_Tp> operator | (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c |= b;
}

typedef Rect_<int> Rect;       // 整形矩形框
typedef Rect_<float> Rect2f;// 浮点数矩形框 亚像素矩形框


// 点结构体
template<typename _Tp>
struct Point_
{
    Point_() : x(0), y(0) {}// 初始化为空
    Point_(_Tp _x, _Tp _y) : x(_x), y(_y) {}// 参数初始化

    _Tp x;
    _Tp y;
};

typedef Point_<int> Point;        // 整数点
typedef Point_<float> Point2f; // 浮点数点 亚像素点

#define CV_8UC1  1    //  Unsigned 8bits 1个通道  灰度图
#define CV_8UC3  3    //  Unsigned 8bits 3个通道  RGB/BGR 彩色图
#define CV_8UC4  4    //  Unsigned 8bits 4个通道  RGBA/BGRA 彩色图
#define CV_32FC1 4    // 32 float 浮点数据，1个通道
// 其他还有 double 64位数据类型CV_64FC1，CV_64FC2，CV_64FC3


// Mat 矩阵结构体
struct Mat
{
    // 数据域指针   被引用计数  行数  列数  通道数
    Mat() : data(0), refcount(0), rows(0), cols(0), c(0) {} // / 初始化为空
    
// 指定行数、列数、通道数，申请存储空间=====================
    Mat(int _rows, int _cols, int flags) : data(0), refcount(0) 
    {
        create(_rows, _cols, flags); //由行数和列数以及通道数，申请数据存储空间
    }
    
    
// 使用对象进行浅拷贝 copy 仅复制数据域指针 参数 且增加引用次数============
    Mat(const Mat& m) : data(m.data), refcount(m.refcount)
    {
        if (refcount)
            NCNN_XADD(refcount, 1);// 引用次数+1

        rows = m.rows;
        cols = m.cols;
        c = m.c;
    }
// 使用参数进行浅拷贝======引用次数怎么设置为0???
    Mat(int _rows, int _cols, int flags, void* _data) : data((unsigned char*)_data), refcount(0)
    {
        rows = _rows;
        cols = _cols;
        c = flags;
    }
    
// 结构体析构函数
    ~Mat()
    {
        release();
    }

// 矩阵 等号 = 运算符 重载=====================
    // assign
    Mat& operator=(const Mat& m)
    {
        if (this == &m) // 同一个对象
            return *this;

        if (m.refcount)
            NCNN_XADD(m.refcount, 1);// 引用次数+1

        release();

        data = m.data; // 数据指针
        refcount = m.refcount;// 被引用计数

        rows = m.rows; // 复制 行、列、通道 参数
        cols = m.cols;
        c = m.c;

        return *this; // 返回该对象
    }

// 由行数和列数以及通道数，申请数据存储空间=====
    void create(int _rows, int _cols, int flags)
    {
        release();

        rows = _rows;
        cols = _cols;
        c = flags;

        if (total() > 0)// 数据总数
        {
            // refcount address must be aligned, so we expand totalsize here
            size_t totalsize = (total() + 3) >> 2 << 2;// 数据 对齐，就是变成 2^n 大小，加上3，除以4，再乘以4
            data = (unsigned char*)ncnn::fastMalloc(totalsize + (int)sizeof(*refcount));// 数据域后紧跟着 被引用计数
            refcount = (int*)(((unsigned char*)data) + totalsize);// 引用计数 数据区域
            *refcount = 1;// 引用计数设置为1
        }
    }

// 清空数据内存空间 并 清零参数=================
    void release()
    {
        if (refcount && NCNN_XADD(refcount, -1) == 1)// 引用次数少
            ncnn::fastFree(data);

        data = 0;

        rows = 0;
        cols = 0;
        c = 0;

        refcount = 0;
     }

// 深拷贝 Mat对象=============================
    Mat clone() const
    {
        if (empty())
            return Mat();

        Mat m(rows, cols, c);// 新建一个 Mat 结构体对象，引用次数为1

        if (total() > 0) // 从源数据域 data 拷贝长度为 w*h*c的大小的数据区域到  新对象的 data
        {
            memcpy(m.data, data, total());
        }

        return m;
    }
// 是否为空======= inline
    bool empty() const { return data == 0 || total() == 0; }

// 通道数======== inline 
    int channels() const { return c; }

// 数据域总数 ==== inline 
    size_t total() const { return cols * rows * c; }

// 中间数据指针传入行数  行通道优先存储
    const unsigned char* ptr(int y) const { return data + y * cols * c; }

    unsigned char* ptr(int y) { return data + y * cols * c; }

// 获取图像矩阵上对应矩形框位置的 roi
    Mat operator()( const Rect& roi ) const
    {
        if (empty())
            return Mat();

        Mat m(roi.height, roi.width, c); // 新建一个 roi Mat对象

        int sy = roi.y; // 矩阵框左上角点 纵坐标y 起始行
        for (int y = 0; y < roi.height; y++)
        {
            const unsigned char* sptr = ptr(sy) + roi.x * c;// 左上角点 在大Mat上对应点的 数据指针
            unsigned char* dptr = m.ptr(y); // 新Mat的 行起始数据 指针
            memcpy(dptr, sptr, roi.width * c); // 复制 对应数据??? 是不是有问题 不同通道数据在内容中如何排列
            sy++;// 行数++
        }

        return m;
    }

    unsigned char* data; // 数据区域 头指针

    // pointer to the reference counter;
    // when points to user-allocated data, the pointer is NULL
    int* refcount; // 本数据 被引用计数

    int rows; //  行数
    int cols;   //  列数
    int c;        //  通道数

};

#define CV_LOAD_IMAGE_GRAYSCALE 1  // 灰度图
#define CV_LOAD_IMAGE_COLOR 3         //  彩色图
// 读取图像=============================
Mat imread(const std::string& path, int flags);
// 保存图像 写图像============================
void imwrite(const std::string& path, const Mat& m);

#if NCNN_PIXEL
// 双线性插值算法 改变大小==================================================
void resize(const Mat& src, Mat& dst, const Size& size, float sw = 0.f, float sh = 0.f, int flags = 0);
#endif // NCNN_PIXEL

} // namespace cv

#endif // NCNN_OPENCV

#endif // NCNN_OPENCV_H
