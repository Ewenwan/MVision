/*
给图像添加边界
我们学习了图像的卷积操作。
一个很自然的问题是如何处理卷积边缘。
当卷积点在图像边界时会发生什么，如何处理这个问题？

大多数用到卷积操作的OpenCV函数都是将给定图像拷贝到另一个轻微变大的图像中，
然后自动填充图像边界(通过下面示例代码中的各种方式)。
这样卷积操作就可以在边界像素安全执行了(填充边界在操作完成后会自动删除)。

本文档将会探讨填充图像边界的两种方法:
    常数边界 BORDER_CONSTANT: 使用常数填充边界 (i.e. 黑色或者 0)
    复制边界 BORDER_REPLICATE: 复制原图中最临近的行或者列。



    装载图像
    由用户决定使用哪种填充方式。有两个选项:
        常数边界: 所有新增边界像素使用一个常数，程序每0.5秒会产生一个随机数更新该常数值。
        复制边界: 复制原图像的边界像素。

    用户可以选择按 ‘c’ 键 (常数边界) 或者 ‘r’ 键 (复制边界)
    当用户按 ‘ESC’ 键，程序退出。


*/
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;
/// 全局变量
Mat src, dst;
int top1, bottom1, left1, right1;
int borderType;
Scalar value;
char* window_name = "copyMakeBorder Demo";
RNG rng(12345);//随机数

// 
int main( int argc, char** argv )
{

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

  /// 使用说明
  printf( "\n \t copyMakeBorder Demo: \n" );
  printf( "\t -------------------- \n" );
  printf( " ** 常数边界 Press 'c' to set the border to a random constant value \n");
  printf( " ** 复制边界 Press 'r' to set the border to be replicated \n");
  printf( " ** 结束     Press 'ESC' to exit the program \n");

  /// 创建显示窗口
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// 初始化输入参数 
  // 初始化边界宽度参数(top, bottom, left 和 right)。我们将它们设定为图像 src 大小的5%
  top1 = (int) (0.05*src.rows); bottom1 = (int) (0.05*src.rows);//上下部 看行
  left1 = (int) (0.05*src.cols); right1 = (int) (0.05*src.cols);//左右部 看列
  dst = src;

  imshow( window_name, dst );//显示图像

  while( true )
    {
      c = waitKey(500);//获取按键

      if( (char)c == 27 )
        { break; }
      else if( (char)c == 'c' )
        { borderType = BORDER_CONSTANT; }//更换 边界参数  常数边界
      else if( (char)c == 'r' )
        { borderType = BORDER_REPLICATE; }//更换 边界参数  复制边界
	// 每个循环 (周期 0.5 秒), 变量 value 自动更新...
	//value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
	//为一个由 RNG 类型变量 rng 产生的随机数。 随机数的范围在 [0,255] 之间。
      value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );//随机像素值 三通道
      copyMakeBorder( src, dst, top1, bottom1, left1, right1, borderType, value );

      imshow( window_name, dst );
    }

  return 0;
}
