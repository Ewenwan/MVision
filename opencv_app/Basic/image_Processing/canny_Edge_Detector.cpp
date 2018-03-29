/*
Canny 边缘检测 边缘检测最优算法
综合使用 高斯平滑 soble梯度检测 非极大值抑制 滞后阈值 等操作来检测物体边缘


Canny 边缘检测算法 是 John F. Canny 于 1986年开发出来的一个多级边缘检测算法，
也被很多人认为是边缘检测的 最优算法, 最优边缘检测的三个主要评价标准是:

        低错误率: 标识出尽可能多的实际边缘，同时尽可能的减少噪声产生的误报。
        高定位性: 标识出的边缘要与图像中的实际边缘尽可能接近。
        最小响应: 图像中的边缘只能标识一次。

步骤：:
 【1】消除噪声。 使用高斯平滑滤波器卷积降噪。 下面显示了一个 size = 5 的高斯内核示例:
	  K = 1/159 [2  4  5  4  2
		     4  9  12 9  4
	 	     5 12  15 12 5
	  	     4  9  12 9  4
		     2  4  5  4  2]
 【2】计算梯度幅值和方向。 
       此处，按照Sobel滤波器的步骤:
    在两个方向求导:
        水平变化: 将 I 与一个奇数大小的内核 G_{x} 进行卷积。比如，当内核大小为3时, G_{x} 的计算结果为:
		G_{x} = [-1 0 +1
			 -2 0 +2
			 -1 0 +1]
        垂直变化: 将:m I 与一个奇数大小的内核 G_{y} 进行卷积。
                  比如，当内核大小为3时, G_{y} 的计算结果为:
		G_{y} = [-1 -2 -1
			  0  0  0
			 +1 +2 +1]
       在图像的每一点，结合以上两个结果求出近似 梯度:
              G = sqrt（GX^2 + GY^2）
              梯度角度方向 = arctan(GY/GX)
              梯度方向近似到四个可能角度之一(一般 0, 45, 90, 135)
  【3】非极大值 抑制。 这一步排除非边缘像素， 仅仅保留了一些细线条(候选边缘)。

  【4】滞后阈值: 最后一步，Canny 使用了滞后阈值，滞后阈值需要两个阈值(高阈值和低阈值):

    如果某一像素位置的幅值超过 高 阈值, 该像素被保留为边缘像素。
    如果某一像素位置的幅值小于 低 阈值, 该像素被排除。
    如果某一像素位置的幅值在两个阈值之间,该像素在连接到一个高于 高阈值的像素时被保留。

Canny 推荐的 高:低 阈值比在 2:1 到3:1之间。


*/
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
using namespace std;
using namespace cv;

/// 全局变量

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;//最大 低阈值
int ratio = 3;// 高低阈值 比值
int kernel_size = 3;//candy 核尺寸
char* window_name = "Edge Map";

// 回调函数 CannyThreshold
//@简介： trackbar 交互回调 - Canny阈值输入比例1:3

void CannyThreshold(int, void*)
{
  /// 使用 3x3内核 均值滤波 降噪
  blur( src_gray, detected_edges, Size(3,3) );

  /// 运行Canny算子
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// 使用 Canny算子输出边缘作为掩码显示原图像
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  imshow( window_name, dst );
 }


/** @函数 main */
int main( int argc, char** argv )
{

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
  /// 创建与src同类型和大小的矩阵(dst)
  dst.create( src.size(), src.type() );

  /// 原图像转换为灰度图像
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// 创建显示窗口
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// 创建trackbar  滑动条 调节 低阈值参数   回调函数 CannyThreshold 
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// 显示图像
  CannyThreshold(0, 0);

  /// 等待用户反应
  waitKey(0);//按键后结束

  return 0;
  }


