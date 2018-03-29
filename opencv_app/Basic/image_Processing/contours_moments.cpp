/*
轮廓矩


因为我们常常会将随机变量（先假定有任意阶矩）作一个线性变换，
把一阶矩（期望）归零，
二阶矩（方差）归一，以便统一研究一些问题。
三阶矩，就是我们所称的「偏度」。
    典型的正偏度投资，就是彩票和保险：
      一般来说，你花的那一点小钱就打水漂了，但是这一点钱完全是在承受范围内的；
       而这点钱则部分转化为小概率情况下的巨大收益。
    而负偏度变量则正好相反，「一般为正，极端值为负」，
      可以参照一些所谓的「灰色产业」：
      一般情况下是可以赚到一点钱的，但是有较小的概率「东窗事发」，赔得血本无归。


四阶矩，又称峰度，简单来说相当于「方差的方差」，
和偏度类似，都可以衡量极端值的情况。峰度较大通常意味着极端值较常出现，
峰度较小通常意味着极端值即使出现了也不会「太极端」。
峰度是大还是小通常与3（即正态分布的峰度）相比较。
至于为什么五阶以上的矩没有专门的称呼，主要是因为我们习惯的线性变换，
只有两个自由度，故最多只能将前两阶矩给「标准化」。
这样，标准化以后，第三、第四阶的矩就比较重要了，前者衡量正负，
后者衡量偏离程度，与均值、方差的关系类似。换句话说，
假如我们能把前四阶矩都给「标准化」了，那么五阶、六阶的矩就会比较重要了吧。



*/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/// 回调函数声明
void thresh_callback(int, void* );

// @主函数 
int main( int argc, char** argv )
{
// 加载源图像
    string imageName("../../common/data/apple.jpeg"); // 图片文件名路径（默认值）
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

  /// 把原图像转化成灰度图像并进行平滑
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  /// 创建新窗口
  char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  // 滑动条 动态改变参数   二值化 阈值边界检测  thresh
  createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
  thresh_callback( 0, 0 );

  waitKey(0);
  return(0);
}

// @thresh_callback 函数 
void thresh_callback(int, void* )
{
  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// 使用Canndy检测边缘  低阈值 thresh 高阈值 thresh*2  核大小
  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
  /// 找到轮廓
  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// 计算矩
  vector<Moments> mu(contours.size() );
  for( int i = 0; i < contours.size(); i++ )
     { mu[i] = moments( contours[i], false ); }

  ///  计算中心矩:
  vector<Point2f> mc( contours.size() );
  for( int i = 0; i < contours.size(); i++ )
     { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

  /// 绘制轮廓
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
       circle( drawing, mc[i], 4, color, -1, 8, 0 );
     }

  /// 显示到窗口中
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );

  /// 通过m00计算轮廓面积并且和OpenCV函数比较
  printf("\t Info: Area and Contour Length \n");
  for( int i = 0; i< contours.size(); i++ )
     {
       printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
       circle( drawing, mc[i], 4, color, -1, 8, 0 );
     }
}
