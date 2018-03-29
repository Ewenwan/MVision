/*
直方图对比
    如何使用OpenCV函数 compareHist 产生一个表达两个直方图的相似度的数值。
    如何使用不同的对比标准来对直方图进行比较。

要比较两个直方图( H_{1} and H_{2} ), 
首先必须要选择一个衡量直方图相似度的 对比标准 (d(H_{1}, H_{2})) 。
OpenCV 函数 compareHist 执行了具体的直方图对比的任务。
该函数提供了4种对比标准来计算相似度：

【1】相关关系 Correlation ( CV_COMP_CORREL )  与均值的偏差 积

【2】平方差  Chi-Square ( CV_COMP_CHISQR )

【3】交集？Intersection ( CV_COMP_INTERSECT ) 对应最小值纸盒

【4】Bhattacharyya 距离( CV_COMP_BHATTACHARYYA ) 巴氏距离（巴塔恰里雅距离 / Bhattacharyya distance）

本程序做什么?

    装载一张 基准图像 和 两张 测试图像 进行对比。
    产生一张取自 基准图像 下半部的图像。
    将图像转换到HSV格式。
    计算所有图像的H-S直方图，并归一化以便对比。
    将 基准图像 直方图与 两张测试图像直方图，基准图像半身像直方图，以及基准图像本身的直方图分别作对比。
    显示计算所得的直方图相似度数值。

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
  Mat src_base, hsv_base;
  Mat src_test1, hsv_test1;
  Mat src_test2, hsv_test2;
  Mat hsv_half_down;

  /// 装载三张背景环境不同的图像
  if( argc < 4 )
    { printf("** Error. Usage: ./compareHist_Demo <image_settings0> <image_setting1> <image_settings2>\n");
      return -1;
    }

  src_base = imread( argv[1], 1 );
  src_test1 = imread( argv[2], 1 );
  src_test2 = imread( argv[3], 1 );

  /// 转换到 HSV
  cvtColor( src_base, hsv_base, CV_BGR2HSV );
  cvtColor( src_test1, hsv_test1, CV_BGR2HSV );
  cvtColor( src_test2, hsv_test2, CV_BGR2HSV );

  hsv_half_down = hsv_base( Range( hsv_base.rows/2, hsv_base.rows - 1 ), Range( 0, hsv_base.cols - 1 ) );

  /// 对hue通道使用30个bin,对saturatoin通道使用32个bin
  int h_bins = 50; int s_bins = 60;
  int histSize[] = { h_bins, s_bins };

  // hue的取值范围从0到256, saturation取值范围从0到180
  float h_ranges[] = { 0, 256 };
  float s_ranges[] = { 0, 180 };

  const float* ranges[] = { h_ranges, s_ranges };

  // 使用第0和第1通道
  int channels[] = { 0, 1 };

  /// 直方图
  MatND hist_base;
  MatND hist_half_down;
  MatND hist_test1;
  MatND hist_test2;

  /// 计算HSV图像的直方图
  calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
  normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

  calcHist( &hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false );
  normalize( hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat() );

  calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
  normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

  calcHist( &hsv_test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false );
  normalize( hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat() );

  ///应用不同的直方图对比方法
  for( int i = 0; i < 4; i++ )
     { int compare_method = i;
       double base_base = compareHist( hist_base, hist_base, compare_method );
       double base_half = compareHist( hist_base, hist_half_down, compare_method );
       double base_test1 = compareHist( hist_base, hist_test1, compare_method );
       double base_test2 = compareHist( hist_base, hist_test2, compare_method );

       printf( " Method [%d] Perfect, Base-Half, Base-Test(1), Base-Test(2) : %f, %f, %f, %f \n", i, base_base, base_half , base_test1, base_test2 );
     }

  printf( "Done \n" );

  return 0;
 }

