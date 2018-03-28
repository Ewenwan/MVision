/*
矩阵掩码操作
利用滤波器核(掩码矩阵) 对矩阵的各个像素重新计算，根据周围像素之间的关系

掩模矩阵控制了旧图像当前位置以及周围位置像素对新图像当前位置像素值的影响力度。
用数学术语讲，即我们自定义一个权重表。

【1】不属于传统的增强对比度，更像是锐化  自己利用 C语言的下标[]访问 实现

【2】内部函数实现（估计利用了多线程，比较快）
     filter2D( 输入Mat I, 输出Mat K, 位深 I.depth(), mask矩阵 kern );// 输入Mat I 输出Mat K 位深，mask矩阵

*/
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

static void help(char* progName)
{
    cout << endl
        <<  "This program shows how to filter images with mask: the write it yourself and the"
        << "filter2d way. " << endl
        <<  "Usage:"                                                                      << endl
        << progName << " [image_name -- default ../data/77.jpg] [G -- grayscale] "        << endl << endl;
}

// 不属于传统的增强对比度，更像是锐化  自己利用 C语言的下标[]访问 实现
// I(i,j)=5∗I(i,j)−[I(i−1,j)+I(i+1,j)+I(i,j−1)+I(i,j+1)]
void Sharpen(const Mat& myImage,Mat& Result);

int main( int argc, char* argv[])
{
    help(argv[0]);
    const char* filename = argc >=2 ? argv[1] : "../data/77.jpeg";

    Mat I, J, K;

    if (argc >= 3 && !strcmp("G", argv[2]))//灰度格式读取
        I = imread( filename, IMREAD_GRAYSCALE);
    else
        I = imread( filename, IMREAD_COLOR);//原图像格式读取

    if (I.empty())
    {
        cout << "The image" << argv[1] << " could not be loaded." << endl;
        return -1;
    }

    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
//=========自己利用 C语言的下标[]访问 实现 盒子滤波=============
    imshow("Input", I);//原图
    double t = (double)getTickCount();
    Sharpen(I, J);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Hand written function times passed in seconds: " << t << endl;
    imshow("Output", J);
    waitKey(0);

//=========内部函数实现（估计利用了多线程，比较快）==========================
    Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
                                   -1,  5, -1,
                                    0, -1,  0);
    t = (double)getTickCount();
    filter2D(I, K, I.depth(), kern );// 输入Mat I 输出Mat K 位深，mask矩阵
// 第五个参数可以设置mask矩阵的中心位置，第六个参数决定在操作不到的地方（边界）如何进行操作。
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Built-in filter2D time passed in seconds:      " << t << endl;
    imshow("Output", K);
    waitKey(0);


    return 0;
}

// 不属于传统的增强对比度，更像是锐化  自己利用 C语言的下标[]访问 实现
// I(i,j)=5∗I(i,j)−[I(i−1,j)+I(i+1,j)+I(i,j−1)+I(i,j+1)]
void Sharpen(const Mat& myImage, Mat& Result)//注意形参为 引用 可以避免拷贝
{
// assert 确保图片格式为 8为无符号 精度，应为下面的访问格式是 8u
// 可以换成其他格式   检查图像位深，如果条件为假则抛出异常
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images

    const int nChannels = myImage.channels();//通道数量 确定每一行究竟有多少子列
    Result.create(myImage.size(),myImage.type());//创建一个新的矩阵

    for(int j = 1 ; j < myImage.rows-1; ++j)//遍历 除去第一行和最后一行的每一行
    {// C语言的下标[]访问
        const uchar* previous = myImage.ptr<uchar>(j - 1);//上一行
        const uchar* current  = myImage.ptr<uchar>(j    );//本行
        const uchar* next     = myImage.ptr<uchar>(j + 1);//下一行

        uchar* output = Result.ptr<uchar>(j);//对应输出矩阵的一行

        for(int i= nChannels;i < nChannels*(myImage.cols-1); ++i)//每一大列
        {
            *output++ = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);//对应通道的像素操作
// saturate_cast 加入了溢出保护  类似下面的操作 8u 上下限是0~255
// if(data<0)  
//         data=0;  
// else if(data>255)  
//     data=255;
        }
    }
// 最下面是边界4行都设置成0；
    Result.row(0).setTo(Scalar(0));
    Result.row(Result.rows-1).setTo(Scalar(0));
    Result.col(0).setTo(Scalar(0));
    Result.col(Result.cols-1).setTo(Scalar(0));
}
