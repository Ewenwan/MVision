/* Harris角点

算法基本思想是使用一个固定窗口在图像上进行任意方向上的滑动，
比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，
如果存在任意方向上的滑动，都有着较大灰度变化，
那么我们可以认为该窗口中存在角点。

图像特征类型:
    边缘 （Edges   物体边缘）
    角点 (Corners  感兴趣关键点（ interest points） 边缘交叉点 )
    斑点(Blobs  感兴趣区域（ regions of interest ） 交叉点形成的区域 )


为什么角点是特殊的?
    因为角点是两个边缘的连接点(交点)它代表了两个边缘变化的方向上的点。
    图像梯度有很高的变化。这种变化是可以用来帮助检测角点的。


G = SUM( W(x,y) * [I(x+u, y+v) -I(x,y)]^2 )

 [u,v]是窗口的偏移量
 (x,y)是窗口内所对应的像素坐标位置，窗口有多大，就有多少个位置
 w(x,y)是窗口函数，最简单情形就是窗口内的所有像素所对应的w权重系数均为1。
                   设定为以窗口中心为原点的二元正态分布

泰勒展开（I(x+u, y+v) 相当于 导数）
G = SUM( W(x,y) * [I(x,y) + u*Ix + v*Iy - I(x,y)]^2)
  = SUM( W(x,y) * (u*u*Ix*Ix + v*v*Iy*Iy))
  = SUM(W(x,y) * [u v] * [Ix^2   Ix*Iy] * [u 
                          Ix*Iy  Iy^2]     v] )
  = [u v]  * SUM(W(x,y) * [Ix^2   Ix*Iy] ) * [u  应为 [u v]为常数 可以拿到求和外面
                           Ix*Iy  Iy^2]      v]    
  = [u v] * M * [u
                 v]
则计算 det(M)   矩阵M的行列式的值  取值为一个标量，写作det(A)或 | A |  矩阵表示的空间的单位面积/体积/..
       trace(M) 矩阵M的迹         矩阵M的对角线元素求和，用字母T来表示这种算子，他的学名叫矩阵的迹

M的两个特征值为 lamd1  lamd2

det(M)    = lamd1 * lamd2
 trace(M) = lamd1 + lamd2

R = det(M)  -  k*(trace(M))^2 
其中k是常量，一般取值为0.04~0.06，
R大于一个阈值的话就认为这个点是 角点

因此可以得出下列结论：
>特征值都比较大时，即窗口中含有角点
>特征值一个较大，一个较小，窗口中含有边缘
>特征值都比较小，窗口处在平坦区域

https://blog.csdn.net/woxincd/article/details/60754658


*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

//全局变量
Mat src, src_gray;
int thresh = 200;//阈值  R大于一个阈值的话就认为这个点是 角点
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";

//函数声明  滑动条回调函数
void cornerHarris_demo( int, void* );

//主函数
int main( int argc, char** argv )
{
  string imageName("../../common/data/building.jpg"); // 图片文件名路径（默认值）
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
  // 转换成灰度图
  cvtColor( src, src_gray, CV_BGR2GRAY );

  //创建一个窗口 和 滑动条
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  createTrackbar( "Threshold阈值: ", source_window, &thresh, max_thresh, cornerHarris_demo );
  imshow( source_window, src );//显示图像

  cornerHarris_demo( 0, 0 );//初始化 滑动条回调函数

  waitKey(0);
  return(0);
}

// 滑动条回调函数
void cornerHarris_demo( int, void* )
{

  Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( src.size(), CV_32FC1 );

  /// 检测参数
  int blockSize = 2;// 滑动窗口大小
  int apertureSize = 3;// Sobel算子的大小(默认值为3) 用来计算 梯度 Ix^2   Ix*Iy Iy^2 
  double k = 0.04;// R = det(M)  -  k*(trace(M))^2   一般取值为0.04~0.06

  /// 检测角点
  cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

  /// 归一化  得到 R值 的 图像大小矩阵
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );

  /// 在检测到的角点处  画圆
  for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )// R大于一个阈值的话就认为这个点是 角点
              {
               circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              }
          }
     }
  /// 显示结果
  namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
  imshow( corners_window, dst_norm_scaled );
}

