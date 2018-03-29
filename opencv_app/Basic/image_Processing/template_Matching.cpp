/*
图像 模板匹配  是一项在一幅图像中寻找与另一幅模板图像最匹配(相似)部分的技术.
使用OpenCV函数 matchTemplate 在模板块和输入图像之间寻找匹配, 获得匹配结果图像
使用OpenCV函数 minMaxLoc 在给定的矩阵(上述得到的匹配结果矩阵)中寻找最大和最小值(包括它们的位置).


什么是模板匹配?
模板匹配是一项在一幅图像中寻找与另一幅模板图像最匹配(相似)部分的技术.
我们需要2幅图像:
    原图像 (I): 在这幅图像里,我们希望找到一块和模板匹配的区域
    模板 (T): 将和原图像比照的图像块
我们的目标是检测最匹配的区域:
为了确定匹配区域, 我们不得不滑动模板图像和原图像进行 比较 :

通过 滑动, 我们的意思是图像块一次移动一个像素 (从左往右,从上往下). 
在每一个位置, 都进行一次度量计算来表明它是 “好” 或 “坏” 地与那个位置匹配 (或者说块图像和原图像的特定区域有多么相似).
对于 T 覆盖在 I 上的每个位置,你把度量值 保存 到 结果图像矩阵 (R) 中. 在 R 中的每个位置 (x,y) 都包含匹配度量值:

上图就是 TM_CCORR_NORMED 方法处理后的结果图像 R . 最白的位置代表最高的匹配. 正如您所见, 红色椭圆框住的位置很可能是结果图像矩阵中的最大数值, 所以这个区域 (以这个点为顶点,长宽和模板图像一样大小的矩阵) 被认为是匹配的.
实际上, 我们使用函数 minMaxLoc 来定位在矩阵 R 中的最大值点 (或者最小值, 根据函数输入的匹配参数) .

OpenCV中支持哪些匹配算法
【1】 平方差匹配 method=CV_TM_SQDIFF  square dirrerence(error)
     这类方法利用平方差来进行匹配,最好匹配为0.匹配越差,匹配值越大.
【2】标准平方差匹配 method=CV_TM_SQDIFF_NORMED  standard  square dirrerence(error)

【3】 相关匹配 method=CV_TM_CCORR
     这类方法采用模板和图像间的乘法操作,所以较大的数表示匹配程度较高,0标识最坏的匹配效果.
【4】 标准相关匹配 method=CV_TM_CCORR_NORMED

【5】 相关匹配 method=CV_TM_CCOEFF
     这类方法将模版对其均值的相对值与图像对其均值的相关值进行匹配,1表示完美匹配,
     -1表示糟糕的匹配,0表示没有任何相关性(随机序列).

【6】标准相关匹配 method=CV_TM_CCOEFF_NORMED

通常,随着从简单的测量(平方差)到更复杂的测量(相关系数),
我们可获得越来越准确的匹配(同时也意味着越来越大的计算代价). 
最好的办法是对所有这些设置多做一些测试实验,
以便为自己的应用选择同时兼顾速度和精度的最佳方案.


在这程序实现了什么?

    载入一幅输入图像和一幅模板图像块 (template)
    通过使用函数 matchTemplate 实现之前所述的6种匹配方法的任一个. 用户可以通过滑动条选取任何一种方法.
    归一化匹配后的输出结果
    定位最匹配的区域
    用矩形标注最匹配的区域


*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/// 全局变量
Mat img; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// 函数声明
void MatchingMethod( int, void* );

// @主函数
int main( int argc, char** argv )
{
  /// 载入原图像和模板块
  img = imread( argv[1], 1 );
  templ = imread( argv[2], 1 );

  /// 创建窗口
  namedWindow( image_window, CV_WINDOW_AUTOSIZE );
  namedWindow( result_window, CV_WINDOW_AUTOSIZE );

  /// 创建滑动条
  char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
  createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );

  MatchingMethod( 0, 0 );

  waitKey(0);
  return 0;
}

/**
 * @函数 MatchingMethod
 * @简单的滑动条回调函数
 */
void MatchingMethod( int, void* )
{
  /// 将被显示的原图像
  Mat img_display;
  img.copyTo( img_display );

  /// 创建输出结果的矩阵 
// 创建了一幅用来存放匹配结果的输出图像矩阵. 仔细看看输出矩阵的大小(它包含了所有可能的匹配位置)
  int result_cols =  img.cols - templ.cols + 1;
  int result_rows = img.rows - templ.rows + 1;
  result.create( result_cols, result_rows, CV_32FC1 );

  /// 进行匹配和标准化
// 很自然地,参数是输入图像 I, 模板图像 T, 结果图像 R 还有匹配方法 (通过滑动条给出)
  matchTemplate( img, templ, result, match_method );
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );//对结果进行归一化:

  /// 通过函数 minMaxLoc 定位最匹配的位置
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
  /// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }

  /// 让我看看您的最终结果
// 源图上显示
  rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
// 匹配结果图上显示
  rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );

  imshow( image_window, img_display );
  imshow( result_window, result );

  return;
}
