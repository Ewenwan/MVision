/*
创建包围轮廓的矩形和圆形边界框
使用Threshold检测边缘  二值化 阈值 thresh
找到轮廓  findContours


*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;// 二值化阈值
int max_thresh = 255;
RNG rng(12345);

/// 函数声明 回调函数
void thresh_callback(int, void* );

// @主函数
int main( int argc, char** argv )
{
/// 加载源图像
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
  /// 转化成灰度图像并进行平滑
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  /// 创建窗口
  char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
  thresh_callback( 0, 0 );

  waitKey(0);
  return(0);
}

/** @thresh_callback 函数 */
void thresh_callback(int, void* )
{
  Mat threshold_output;//阈值检测边缘
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// 使用Threshold检测边缘  二值化 阈值 thresh
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  /// 找到轮廓  轮廓  contours
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// 多边形逼近轮廓 + 获取矩形和圆形边界框
  vector<vector<Point> > contours_poly( contours.size() );// 多边形逼近轮廓
  vector<Rect> boundRect( contours.size() );//矩形边界框
  vector<Point2f>center( contours.size() );//圆形边界框 中心
  vector<float>radius( contours.size() );//圆形边界框 半径

  for( int i = 0; i < contours.size(); i++ )//对于每一个轮廓
     { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );// 多边形逼近轮廓
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );//矩形边界框
       minEnclosingCircle( contours_poly[i], center[i], radius[i] );//圆形边界框
     }


  /// 画多边形轮廓 + 包围的矩形框 + 圆形框
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );//多边形轮廓 
       rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );//矩形框
       circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );//圆形框
     }

  /// 显示在一个窗口
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}
