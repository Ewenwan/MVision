/*

为轮廓创建可倾斜的边界框和椭圆


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

/// 回调函数Function header
void thresh_callback(int, void* );

// @function main 
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

  /// 转为灰度图并模糊化
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  /// 创建窗体
  char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  // 滑动条 动态改变参数   二值化 阈值边界检测  thresh
  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
  thresh_callback( 0, 0 );

  waitKey(0);
  return(0);
}

// @function thresh_callback 
void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// 阈值化检测边界 二值化 阈值边界检测
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  /// 寻找轮廓
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// 对每个找到的轮廓创建可倾斜的边界框和椭圆
  vector<RotatedRect> minRect( contours.size() );// 可倾斜的边界框
  vector<RotatedRect> minEllipse( contours.size() );// 椭圆

  for( int i = 0; i < contours.size(); i++ )
     { minRect[i] = minAreaRect( Mat(contours[i]) );// 可倾斜的边界框
       if( contours[i].size() > 5 )
         { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }// 椭圆
     }

  /// 绘出轮廓及其可倾斜的边界框和边界椭圆
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       // contour 轮廓
       drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       // ellipse 椭圆
       ellipse( drawing, minEllipse[i], color, 2, 8 );
       // rotated rectangle 可倾斜的边界框
       Point2f rect_points[4]; minRect[i].points( rect_points );
       for( int j = 0; j < 4; j++ )
          line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
     }

  /// 结果在窗体中显示
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}
