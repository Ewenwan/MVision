/*
Sobel 算子x方向、y方向导数 合成导数
一个最重要的卷积运算就是导数的计算(或者近似计算).
为什么对图像进行求导是重要的呢? 假设我们需要检测图像中的 边缘 球图像梯度大的地方

【1】Sobel算子
	Sobel 算子是一个离散微分算子 (discrete differentiation operator)。 
	      它用来计算图像灰度函数的近似梯度。
	Sobel 算子结合了高斯平滑和微分求导。

假设被作用图像为 I:
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
             sqrt（GX^2 + GY^2）


Sobel内核

当内核大小为 3 时, 以上Sobel内核可能产生比较明显的误差(毕竟，Sobel算子只是求取了导数的近似值)。
 为解决这一问题，OpenCV提供了 Scharr 函数，但该函数仅作用于大小为3的内核。
该函数的运算与Sobel函数一样快，但结果却更加精确，其内核为:

		G_{x} = [-3  0 +3
			 -10 0 +10
			 -3  0 +3]
		G_{y} = [-3 -10 -3
			  0  0  0
			 +3 +10 +3]

*/


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;
using namespace cv;

/** @function main */
int main( int argc, char** argv )
{

  Mat src, src_gray;
  Mat grad;
  char* window_name = "Sobel Demo - Simple Edge Detector";
  int scale = 1;// 计算导数 放大因子 scale ？
  int delta = 0;// 偏置 delta: 在卷积过程中，该值会加到每个像素上。默认情况下，这个值为 0 
  int ddepth = CV_16S;// ddepth: 输出图像的深度，设定为 CV_16S 避免外溢。

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
  // 高斯平滑  降噪 ( 内核大小 = 3 )
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  // 将降噪后的图像转换为灰度图:
  cvtColor( src, src_gray, CV_RGB2GRAY );

  /// 创建显示窗口
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// 创建 水平和垂直梯度图像 grad_x 和 grad_y 
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// 求 X方向梯度
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
// ddepth: 输出图像的深度，设定为 CV_16S 避免外溢。
// 偏置 delta: 在卷积过程中，该值会加到每个像素上。默认情况下，这个值为 0 
// 计算导数 放大因子 scale ？

  /// 求Y方向梯度
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );
 
  /// 合并梯度(近似)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  imshow( window_name, grad );

  waitKey(0);

  return 0;
  }



