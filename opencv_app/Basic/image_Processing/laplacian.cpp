/*
Laplacian 算子 的离散模拟。 图像二阶倒数  梯度的梯度 0值的话 边缘概率较大

Sobel 算子 ，其基础来自于一个事实，即在边缘部分，像素值出现”跳跃“或者较大的变化。
如果在此边缘部分求取一阶导数，你会看到极值的出现。

你会发现在一阶导数的极值位置，二阶导数为0。
所以我们也可以用这个特点来作为检测图像边缘的方法。
 但是， 二阶导数的0值不仅仅出现在边缘(它们也可能出现在无意义的位置),
但是我们可以过滤掉这些点。


Laplacian 算子

    从以上分析中，我们推论二阶导数可以用来 检测边缘 。 
    因为图像是 “2维”, 我们需要在两个方向求导。使用Laplacian算子将会使求导过程变得简单。
    Laplacian 算子 的定义:

    Laplace(f) = \dfrac{\partial^{2} f}{\partial x^{2}} + \dfrac{\partial^{2} f}{\partial y^{2}}

    OpenCV函数 Laplacian 实现了Laplacian算子。 
    实际上，由于 Laplacian使用了图像梯度，它内部调用了 Sobel 算子。

*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;
/** @函数 main */
int main( int argc, char** argv )
{
  Mat src, src_gray, dst;
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  char* window_name = "Laplace Demo";

  int c;

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


  /// 使用高斯滤波消除噪声
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// 转换为灰度图
  cvtColor( src, src_gray, CV_RGB2GRAY );

  /// 创建显示窗口
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// 使用Laplace函数
  Mat abs_dst;

  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, abs_dst );
// ddepth: 输出图像的深度，设定为 CV_16S 避免外溢。
// kernel_size 卷积核大小
// 偏置 delta: 在卷积过程中，该值会加到每个像素上。默认情况下，这个值为 0 
// 计算导数 放大因子 scale ？
// 边界填充 BORDER_DEFAULT 默认使用 复制填充

  /// 显示结果
  imshow( window_name, abs_dst );

  waitKey(0);

  return 0;
}




