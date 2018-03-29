/*
形态学操作就是基于形状的一系列图像处理操作。
通过将 结构元素 作用于输入图像来产生输出图像。

最基本的形态学操作有二：
腐蚀与膨胀(Erosion 与 Dilation)。 

他们的运用广泛:
    消除噪声
    分割(isolate)独立的图像元素，以及连接(join)相邻的元素。
    寻找图像中的明显的极大值区域或极小值区域。 连通域

【1】膨胀Dilation
选择核内部的最大值(值越大越亮 约白)

此操作将图像 A 与任意形状的内核 (B)，通常为正方形或圆形,进行卷积。
内核 B 有一个可定义的 锚点, 通常定义为内核中心点。
进行膨胀操作时，将内核 B 划过图像,将内核 B 覆盖区域的最大相素值提取，
并代替锚点位置的相素。显然，这一最大化操作将会导致图像中的亮区开始”扩展” 
(因此有了术语膨胀 dilation )。

背景(白色)膨胀，而黑色字母缩小了。


【2】腐蚀 Erosion 
选择核内部的最小值(值越小越暗 约黑)


腐蚀在形态学操作家族里是膨胀操作的孪生姐妹。它提取的是内核覆盖下的相素最小值。
进行腐蚀操作时，将内核 B 划过图像,将内核 B 覆盖区域的最小相素值提取，并代替锚点位置的相素。
以与膨胀相同的图像作为样本,我们使用腐蚀操作。

从下面的结果图我们看到亮区(背景)变细，而黑色区域(字母)则变大了。


*/
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

// 全局变量
Mat src, erosion_dst, dilation_dst;
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

//窗口名字
string Erosion_w("Erosion 腐蚀 Demo");
string Dilation_w("Dilation 膨胀 Demo");

//函数声明
void Erosion( int, void* );
void Dilation( int, void* );

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

  namedWindow( Erosion_w, WINDOW_AUTOSIZE );
  namedWindow( Dilation_w, WINDOW_AUTOSIZE );
  moveWindow( Dilation_w, src.cols, 0 );//新建一个

// 创建腐蚀 Trackbar
  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", Erosion_w,
          &erosion_elem, max_elem,// 滑动条 动态改变参数 erosion_elem 核窗口形状
          Erosion );//回调函数  Erosion 

  createTrackbar( "Kernel size:\n 2n +1", Erosion_w,
          &erosion_size, max_kernel_size,// 滑动条 动态改变参数 erosion_size 窗口大小
          Erosion );//回调函数  Erosion 


// 创建膨胀 Trackbar
  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", Dilation_w,
          &dilation_elem, max_elem,// 滑动条 动态改变参数 dilation_elem 核窗口形状
          Dilation );//回调函数 Dilation

  createTrackbar( "Kernel size:\n 2n +1", Dilation_w,
          &dilation_size, max_kernel_size,// 滑动条 动态改变参数 dilation_size 窗口大小
          Dilation );//回调函数 Dilation

// 默认 开始参数   长方形核 1核子大小
  Erosion( 0, 0 );
  Dilation( 0, 0 );
  waitKey(0);//等待按键
  return 0;
}

// 腐蚀操作 
void Erosion( int, void* )
{
  int erosion_type = 0;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }// 矩形
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }// 交叉形
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }// 椭圆形
  Mat element = getStructuringElement( erosion_type,//核形状
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),//核大小
                       Point( erosion_size, erosion_size ) );//锚点 默认锚点在内核中心位置
  erode( src, erosion_dst, element );
  imshow( Erosion_w, erosion_dst );
}


// 膨胀操作
void Dilation( int, void* )
{
  int dilation_type = 0;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }// 矩形
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }// 交叉形
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }// 椭圆形
  Mat element = getStructuringElement( dilation_type,//核形状
                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),//核大小
                       Point( dilation_size, dilation_size ) );//锚点 默认锚点在内核中心位置
  dilate( src, dilation_dst, element );
  imshow( Dilation_w, dilation_dst );
}

