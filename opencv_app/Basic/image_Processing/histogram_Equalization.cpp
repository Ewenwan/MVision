/*
求得对直方图均衡化的映射矩阵 在对原图像进行映射
图像的直方图是什么?
    直方图是图像中像素强度分布的图形表达方式.
    它统计了每一个强度值（灰度 0~255 256个值）所具有的像素点个数.


直方图均衡化是什么?

    直方图均衡化是通过拉伸像素强度分布范围来增强图像对比度的一种方法.
    说得更清楚一些, 以上面的直方图为例, 你可以看到像素主要集中在中间的一些强度值上.
 直方图均衡化要做的就是 拉伸 这个范围. 见下面左图: 绿圈圈出了 少有像素分布其上的 强度值. 
对其应用均衡化后, 得到了中间图所示的直方图. 均衡化的图像见下面右图.

直方图均衡化是怎样做到的?
    均衡化指的是把一个分布 (给定的直方图) 映射 
    到另一个分布 (一个更宽更统一的强度值分布), 所以强度值分布会在整个范围内展开.
    要想实现均衡化的效果, 映射函数应该是一个 累积分布函数 (cdf) 


*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

// 主函数
int main( int argc, char** argv )
{
  Mat src, dst;

  char* source_window = "Source image";
  char* equalized_window = "Equalized Image";

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

  /// 转为灰度图
  cvtColor( src, src, CV_BGR2GRAY );

  /// 应用直方图均衡化
  equalizeHist( src, dst );

  /// 显示结果
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  namedWindow( equalized_window, CV_WINDOW_AUTOSIZE );

  imshow( source_window, src );
  imshow( equalized_window, dst );

  /// 等待用户按键退出程序
  waitKey(0);

  return 0;
}


