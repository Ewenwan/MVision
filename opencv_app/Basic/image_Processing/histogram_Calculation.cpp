/*
图像的直方图计算
单个通道内的像素值进行统计 


什么是直方图?
    直方图是对数据的集合 统计 ，并将统计结果分布于一系列预定义的 bins 中。
    这里的 数据 不仅仅指的是灰度值 (如上一篇您所看到的)，
     统计数据可能是任何能有效描述图像的特征。
    先看一个例子吧。 假设有一个矩阵包含一张图像的信息 (灰度值 0-255):

OpenCV提供了一个简单的计算数组集(通常是图像或分割后的通道)
的直方图函数 calcHist 。 支持高达 32 维的直方图。


*/


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** @函数 main */
int main( int argc, char** argv )
{
  Mat src, dst;

 /// 加载源图像
   string imageName("../../common/data/77.jpeg"); // 图片文件名路径（默认值）
    if( argc > 1)
    {
        imageName = argv[1];//如果传递了文件 就更新
    }

   src = imread( imageName );
   if( src.empty() )
    { 
        cout <<  "can't load image " << endl;
	return -1;  
    }


 /// split分割成3个单通道图像 ( R, G 和 B )
 vector<Mat> rgb_planes;
 split( src, rgb_planes );

 /// 设定bin数目
 int histSize = 255;

 /// 设定取值范围 ( R,G,B) )
 float range[] = { 0, 255 } ;
 const float* histRange = { range };

 bool uniform = true; bool accumulate = false;

 Mat r_hist, g_hist, b_hist;

 /// 计算直方图:
 calcHist( &rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
 calcHist( &rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
 calcHist( &rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
// 1: 输入数组的个数 (这里我们使用了一个单通道图像，我们也可以输入数组集 )
// 0: 需要统计的通道 (dim)索引 ，这里我们只是统计了灰度 (且每个数组都是单通道)所以只要写 0 就行了。
// Mat(): 掩码( 0 表示忽略该像素)， 如果未定义，则不使用
// r_hist: 储存直方图的矩阵
// 1: 直方图维数
// histSize: 每个维度的bin数目
// histRange: 每个维度的取值范围
// uniform 和 accumulate: bin大小相同，清除直方图痕迹

 // 创建直方图画布
 int hist_w = 400; int hist_h = 400;
 int bin_w = cvRound( (double) hist_w/histSize );

 Mat histImage( hist_w, hist_h, CV_8UC3, Scalar( 0,0,0) );

 /// 将直方图归一化到范围 [ 0, histImage.rows ]
 normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
 normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
 normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
// 0 及 histImage.rows: 这里，它们是归一化 r_hist 之后的取值极限

 /// 在直方图画布上画出直方图
 for( int i = 1; i < histSize; i++ )
   {
     line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                      Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                      Scalar( 0, 0, 255), 2, 8, 0  );
     line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                      Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                      Scalar( 0, 255, 0), 2, 8, 0  );
     line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                      Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                      Scalar( 255, 0, 0), 2, 8, 0  );
    }

 /// 显示直方图
 namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
 imshow("calcHist Demo", histImage );

 waitKey(0);

 return 0;

}
