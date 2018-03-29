/* 计算物体的凸包  边缘包围圈
对图像进行二值化     candy边缘检测也得到 二值图
寻找轮廓 
对每个轮廓计算其凸包
绘出轮廓及其凸包

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

 /// Function header
 void thresh_callback(int, void* );

/** @function main */
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
   /// 转成灰度图并进行模糊降噪
   cvtColor( src, src_gray, CV_BGR2GRAY );
   blur( src_gray, src_gray, Size(3,3) );

   /// 创建窗体
   char* source_window = "Source";
   namedWindow( source_window, CV_WINDOW_AUTOSIZE );
   imshow( source_window, src );

   createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
   thresh_callback( 0, 0 );

   waitKey(0);
   return(0);
 }


// 回调函数
 void thresh_callback(int, void* )
 {
   Mat src_copy = src.clone();
   Mat threshold_output;
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;

   /// 对图像进行二值化  这里 candy边缘检测也得到 二值图
   threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );

   /// 寻找轮廓
   findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

   /// 对每个轮廓计算其凸包
   vector<vector<Point> >hull( contours.size() );
   for( int i = 0; i < contours.size(); i++ )
      {  convexHull( Mat(contours[i]), hull[i], false ); }

   /// 绘出轮廓及其凸包
   Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
   for( int i = 0; i< contours.size(); i++ )
      {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
      }

   /// 把结果显示在窗体
   namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
   imshow( "Hull demo", drawing );
 }



