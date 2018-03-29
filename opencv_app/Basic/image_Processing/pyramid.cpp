/*
图像金字塔
使用OpenCV函数 pyrUp 和 pyrDown 对图像进行向上和向下采样。
然后高斯平滑

当我们需要将图像转换到另一个尺寸的时候， 有两种可能：

    放大 图像 或者
    缩小 图像。
我们首先学习一下使用 图像金字塔 来做图像缩放, 图像金字塔是视觉运用中广泛采用的一项技术。

图像金字塔：
    一个图像金字塔是一系列图像的集合 - 
所有图像来源于同一张原始图像 - 通过梯次向下采样获得，直到达到某个终止条件才停止采样。

有两种类型的图像金字塔常常出现在文献和应用中:
   【1】 高斯金字塔(Gaussian pyramid): 用来向下采样
   【2】 拉普拉斯金字塔(Laplacian pyramid): 用来从金字塔低层图像重建上层未采样图像
在这篇文档中我们将使用 高斯金字塔 。

高斯金字塔：想想金字塔为一层一层的图像，层级越高，图像越小。

高斯内核:
1/16 [1  4  6  4  1
      4  16 24 16 4
      6  24 36 24 6
      4  16 24 16 4
      1  4  6  4  1]

下采样：
    将 图像 与高斯内核做卷积 
    将所有偶数行和列去除。
    显而易见，结果图像只有原图的四分之一。

如果将图像变大呢?:
    首先，将图像在每个方向扩大为原来的两倍，新增的行和列以0填充(0)
    使用先前同样的内核(乘以4)与放大后的图像卷积，获得 “新增像素” 的近似值。

这两个步骤(向下和向上采样) 分别通过OpenCV函数 pyrUp 和 pyrDown 实现, 

*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// 全局变量
Mat src, dst, tmp;
char* window_name = "Pyramids Demo";


// 主函数
int main( int argc, char** argv )
{
  /// 指示说明
  printf( "\n Zoom In-Out demo  \n " );
  printf( "------------------ \n" );
  printf( " * [u] -> Zoom in  \n" );
  printf( " * [d] -> Zoom out \n" );
  printf( " * [ESC] -> Close program \n \n" );

  /// 测试图像 - 尺寸必须能被 2^{n} 整除
  src = imread("../../common/data/chicky_512.png");
  if( !src.data )
    { printf(" No data! -- Exiting the program \n");
      return -1; }

  tmp = src;
  dst = tmp;

  /// 创建显示窗口
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  imshow( window_name, dst );

  /// 循环 检测 按键响应
  while( true )
  {
    int c;
    c = waitKey(10);//获得按键

    if( (char)c == 27 )//esc
      { break; }
    if( (char)c == 'u' )//上采样
      { pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 ) );
        printf( "** Zoom In: Image x 2 \n" );
      }
    else if( (char)c == 'd' )//下采样
     { pyrDown( tmp, dst, Size( tmp.cols/2, tmp.rows/2 ) );
       printf( "** Zoom Out: Image / 2 \n" );
     }

    imshow( window_name, dst );
    tmp = dst;
  }
  return 0;
}

