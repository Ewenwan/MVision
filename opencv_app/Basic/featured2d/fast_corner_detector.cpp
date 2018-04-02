/* FAST角点检测算法 

FAST角点检测算法  ORB特征检测中使用的就是这种角点检测算法
	周围区域灰度值 都较大 或 较小

        若某像素与其周围邻域内足够多的像素点相差较大，则该像素可能是角点。

	该算法检测的角点定义为在像素点的周围邻域内有足够多的像素点与该点处于不同的区域。
	应用到灰度图像中，即有足够多的像素点的灰度值大于该点的灰度值或者小于该点的灰度值。

	p点附近半径为3的圆环上的16个点，
	一个思路是若其中有连续的12( （FAST-9，当然FAST-10、FAST-11、FAST-12、FAST-12）
        )个点的灰度值与p点的灰度值差别超过某一阈值，
	则可以认为p点为角点。

        之后可进行非极大值抑制

	这一思路可以使用机器学习的方法进行加速。
	对同一类图像，例如同一场景的图像，可以在16个方向上进行训练，
	得到一棵决策树，从而在判定某一像素点是否为角点时，
	不再需要对所有方向进行检测，
	而只需要按照决策树指定的方向进行2-3次判定即可确定该点是否为角点。


*/

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d/features2d.hpp>  
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

//全局变量
Mat src, src_gray;
int thresh = 50;//阈值  R大于一个阈值的话就认为这个点是 角点
int max_thresh = 200;

char* source_window = "Source image";
char* corners_window = "Corners detected";

//函数声明  滑动条回调函数
void cornerFast_demo( int, void* );

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
  createTrackbar( "Threshold阈值: ", source_window, &thresh, max_thresh, cornerFast_demo );
  imshow( source_window, src );//显示图像

  cornerFast_demo( 0, 0 );//初始化 滑动条回调函数

  waitKey(0);
  return(0);
}

// 滑动条回调函数
void cornerFast_demo( int, void* )
{
  Mat dst = src.clone();
  //cv::FastFeatureDetector fast(50);   // 检测的阈值为50  
  std::vector<KeyPoint> keyPoints; 
  //fast.detect(src_gray, keyPoints);  // 检测角点
  FAST(src_gray, keyPoints,thresh);
  // 画角点
  drawKeypoints( dst, keyPoints, dst, Scalar(0,0,255), DrawMatchesFlags::DRAW_OVER_OUTIMG); 
  // 显示结果
  namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
  imshow( corners_window,  dst );
}

