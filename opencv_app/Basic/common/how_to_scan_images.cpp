/*
opencv 多通道图像  是一个二维数组
B的第一列 G的第一列 R的第一列 B的第二列 G的第二列 R的第二列 ...
访问图片的每一个像素值  查找表阶梯化像素值（色域缩减:）
https://blog.csdn.net/baidu_19069751/article/details/50869561


用法
./how_to_scan_images imageName.jpg intValueToReduce [G]
./scan_images ../data/77.jpeg 10 
G选项 可选 表示灰度格式处理

色域缩减: 我们可以把当前的像素值用一个值来划分，比如0-9的像素都归为0，10-19的像素都归为10，依此类推。 
Inew  =  Iold / 10  * 10

这里的10 就是间隔步长 就是 intValueToReduce 参数

【1】C语言的下标[]访问
【2】迭代器访问 MatIterator_<> it, end;
【3】at<>()随机访问
【4】内置LUT函数访问 方式效率最高
内置LUT  方式效率最高
这个函数通过Intel Threaded Building Blocks激活了多线程。
如果你真的需要通过指针的方式去遍历图像，迭代器方式是最安全的，虽然它很慢。

*/
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

static void help()
{
    cout
        << "\n--------------------------------------------------------------------------" << endl
        << "This program shows how to scan image objects in OpenCV (cv::Mat). As use case"
        << " we take an input image and divide the native color palette (255) with the "  << endl
        << "input. Shows C operator[] method, iterators and at function for on-the-fly item address calculation."<< endl
        << "Usage:"                                                                       << endl
        << "./howToScanImages imageNameToUse divideWith [G]"                              << endl
        << "if you add a G parameter the image is processed in gray scale"                << endl
        << "--------------------------------------------------------------------------"   << endl
        << endl;
}

// 【1】C语言的下标[]访问   存储是连续区域
Mat& ScanImageAndReduceC(Mat& I, const uchar* table);

//【2】迭代器访问 MatIterator_<> it, end;
Mat& ScanImageAndReduceIterator(Mat& I, const uchar* table);

//【3】at<>()随机访问
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar * table);

int main( int argc, char* argv[])
{
    help();
    if (argc < 3)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }
//====================【1】参数读取=======================
    Mat I, J;
    if( argc == 4 && !strcmp(argv[3],"G") )//灰度格式读取
        I = imread(argv[1], IMREAD_GRAYSCALE);
    else
        I = imread(argv[1], IMREAD_COLOR);//原图像格式读取   BGR color space

    if (I.empty())
    {
        cout << "The image" << argv[1] << " could not be loaded." << endl;
        return -1;
    }

//=====! [dividewith] 按照分割步长预先计算好查找表=======
//0-9像素值都为0
//10-19像素值都为10
///...
    int divideWith = 0; // convert our input string to number - C++ style
    stringstream s;
    s << argv[2];//字符串 
    s >> divideWith;//使用了C++ stringstream 类进行字符串与int转换
    if (!s || !divideWith)
    {
        cout << "Invalid number entered for dividing. " << endl;
        return -1;
    }
    uchar table[256];
    for (int i = 0; i < 256; ++i)
       table[i] = (uchar)(divideWith * (i/divideWith));//计算查找表（整数除法）
    //! [dividewith]

// =====C语言的下标[]访问 前提是 存储是连续区域===================
    const int times = 100;//100次访问图像的所有像素
    double t;
    t = (double)getTickCount();//当前时间 返回调用此函数时的系统CPU计数
    for (int i = 0; i < times; ++i)
    {
        cv::Mat clone_i = I.clone();
        J = ScanImageAndReduceC(clone_i, table);
    }
// getTickFrequency() 返回每秒钟的CPU计数频  总计数差值/频率 = 时间跨度
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    t /= times;

    cout << "Time of reducing with the C operator [] (averaged for "
         << times << " runs): " << t << " milliseconds."<< endl;

//=================
    t = (double)getTickCount();
    for (int i = 0; i < times; ++i)
    {
        cv::Mat clone_i = I.clone();
        J = ScanImageAndReduceIterator(clone_i, table);
    }
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    t /= times;
    cout << "Time of reducing with the iterator (averaged for "
        << times << " runs): " << t << " milliseconds."<< endl;

//==========即时地址访问 .at<>(,)　不推荐使用如下这种方法=====================
// 你必须不断的获取每一行的首地址然后用[]操作符去获取你要访问的那一列。
    t = (double)getTickCount();
    for (int i = 0; i < times; ++i)
    {
        cv::Mat clone_i = I.clone();
        ScanImageAndReduceRandomAccess(clone_i, table);
    }
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    t /= times;
    cout << "Time of reducing with the on-the-fly address generation - at function (averaged for "
        << times << " runs): " << t << " milliseconds."<< endl;

//======福利—-core function=====================================
// 有一种更简便，输入量更少的方法来实现同样的目的。LUT()函数：
    //! [table-init]
    Mat lookUpTable(1, 256, CV_8U);//创建新的查找表
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = table[i];//更新元素值
    //! [table-init]
    t = (double)getTickCount();
    for (int i = 0; i < times; ++i)
        //! [table-use]
        LUT(I, lookUpTable, J);// (I是输入Mat J是输出Mat)：  图像被 查找表 阶梯化
        //! [table-use]
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    t /= times;
    cout << "Time of reducing with the LUT function (averaged for "
        << times << " runs): " << t << " milliseconds."<< endl;


    return 0;
}

//! [scan-c]
//====C语言的下标[]访问   存储连续区域(所有元素排成一行（内存中）)====================
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
// assert 确保图片格式为 8为无符号 精度，应为下面的访问格式是 8u
// 可以换成其他格式 
    CV_Assert(I.depth() == CV_8U);

    int channels = I.channels();//通道数量

    int nRows = I.rows;//行数
    int nCols = I.cols * channels;//总列数 为列数×通道数 （多通道存储方式）
// 判断是否连续存储
    if (I.isContinuous())
    {
        nCols *= nRows;//列数扩大
        nRows = 1;//存储连续区域(所有元素排成一行（内存中）)
// 非连续的话 行与行之间是有存储间隙的
    }

    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)//每一行
    {// uchar 8u 位无符号格式访问  float 32位  double 64位
        p = I.ptr<uchar>(i);//每一行的数组首地址 
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];//p[j]为原来像素值（0~255）对应到 查找表中 阶梯像素值  替换
        }
    }
    return I;
// 还有另外一种处理方式，Ｍat类的data成员指示了第一行第一列的地址，如果是NULL，
//则说明图片没有正确加载，检查这个指针是否为NULL是最简单的检查图片是否加载成功的方法。
//在连续存储的情况下，我们可以这样写遍历(灰阶图片单通道)：
//uchar* p = I.data;//首地址
//for( unsigned int i =0; i < ncol*nrows; ++i)// 多通道的话 i < ncol*nrows*I.channels()
//  *p++ = table[*p];//取地址中的像素值 查找表替换 更新  指针前移一位


}
//! [scan-c]


// ====安全（迭代器）遍历方式=========
//在前面介绍的高效遍历方式中，传递正确的除数，处理行与行之间的地址间隙都是你的任务，
//安全（迭代器）遍历方式将从你手中接管这些任务，你只需要获取矩阵的开始位置与结束位置，然后递增开始位置迭代器逐个访问。

//! [scan-iterator]
Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table)
{
// assert 确保图片格式为 8为无符号 精度，应为下面的访问格式是 8u
// 可以换成其他格式 
    CV_Assert(I.depth() == CV_8U);

    const int channels = I.channels();//通道数量
    switch(channels)
    {
    case 1://1通道  灰度图
        {
            MatIterator_<uchar> it, end;//每一大列 就一个灰度分量
            for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
                *it = table[*it];//迭代器存储的是地址 *it得到像素值
            break;
        }
    case 3://3通道彩色图
        {
            MatIterator_<Vec3b> it, end;//每一大列 具有 BRG三个通道的分量 Vec3b
// 如果你对彩色图片使用一个简单的uchar迭代器，那么你将只能访问B通道。 只获得 一大列的第一个元素 B通道的像素值
            for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
            {
                (*it)[0] = table[(*it)[0]];
                (*it)[1] = table[(*it)[1]];
                (*it)[2] = table[(*it)[2]];
            }
        }
    }

    return I;
}

//! [scan-iterator]


//======即时地址访问 .at<>(,)　不推荐使用如下这种方法=============================
// 你必须不断的获取每一行的首地址然后用[]操作符去获取你要访问的那一列。 
//! [scan-random]
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
{

// assert 确保图片格式为 8为无符号 精度，应为下面的访问格式是 8u
// 可以换成其他格式 
    CV_Assert(I.depth() == CV_8U);

    const int channels = I.channels();//通道数量
    switch(channels)
    {
    case 1://1通道  灰度图
        {
            for( int i = 0; i < I.rows; ++i)
                for( int j = 0; j < I.cols; ++j )
                    I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];//I.at<uchar>(i,j)为原始像素值
            break;
        }
    case 3://3通道彩色图
        {
// 需要倍乘lookup table，使用at()将会十分的繁琐，需要不断输入数据类型和关键字。opencv引入了Mat_类型来解决这个问题
         Mat_<Vec3b> _I = I;// Mat 转换成 Mat_

         for( int i = 0; i < I.rows; ++i)//每一行
            for( int j = 0; j < I.cols; ++j )//每一大列（对于彩色图来说是 3个小列  BGR）
               {
                   _I(i,j)[0] = table[_I(i,j)[0]];
                   _I(i,j)[1] = table[_I(i,j)[1]];
                   _I(i,j)[2] = table[_I(i,j)[2]];
            }
         I = _I;//Mat_ 和Mat可以方便的互相转换 
         break;
        }
    }

    return I;
}
//! [scan-random]
