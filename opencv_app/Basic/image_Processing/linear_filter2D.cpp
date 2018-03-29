/*
自定义滤波器核
用OpenCV函数 filter2D 创建自己的线性滤
      kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

【1】卷积
高度概括地说，卷积是在每一个图像块与某个算子（核）之间进行的运算。

【2】核是什么？
核说白了就是一个固定大小的数值数组。该数组带有一个 锚点 ，一般位于数组中央。


*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;
//主函数
int main ( int argc, char** argv )
{
  /// 声明变量
  Mat src, dst;

  Mat kernel;
  Point anchor;
  double delta;
  int ddepth;
  int kernel_size;
  char* window_name = "filter2D Demo";

  int c;

    string imageName("../../common/data/notes.png"); // 图片文件名路径（默认值）
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
  /// 创建窗口
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// 初始化滤波器参数
  anchor = Point( -1, -1 );//锚点
  delta = 0;// 偏置 delta: 在卷积过程中，该值会加到每个像素上。默认情况下，这个值为 0 
  ddepth = -1;
// ddepth: dst 的深度。若为负值（如 -1 ），则表示其深度与源图像相等。

  /// 循环 - 每隔0.5秒，用一个不同的核来对图像进行滤波
  int ind = 0;
  while( true )
    {
      c = waitKey(500);
      /// 按'ESC'可退出程序
      if( (char)c == 27 )
        { break; }

      /// 更新归一化块滤波器的核大小
      kernel_size = 3 + 2*( ind%5 );//ind%5  0，1，2,3,4
      // 核的大小 设置为 [3,11] 范围内的奇数
      // 第二行代码把1填充进矩阵，并执行归一化——除以矩阵元素数——以构造出所用的核。
      kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

      /// 使用滤波器
// ddepth: dst 的深度。若为负值（如 -1 ），则表示其深度与源图像相等。
// delta: 在卷积过程中，该值会加到每个像素上。默认情况下，这个值为 0 
      filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
      imshow( window_name, dst );
      ind++;//核子尺寸参数
    }

  return 0;
}
