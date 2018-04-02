/*  cornerSubPix()
亚像素级的角点检测
使用OpenCV函数 cornerSubPix 
寻找更精确的角点位置 (不是整数类型的位置，而是更精确的浮点类型位置).


除了利用Harris进行角点检测和利用Shi-Tomasi方法进行角点检测外，
还可以使用cornerEigenValsAndVecs()函数和cornerMinEigenVal()函数自定义角点检测函数。
如果对角点的精度有更高的要求，可以用cornerSubPix()函数将角点定位到子像素，从而取得亚像素级别的角点检测效果。


使用cornerSubPix()函数在goodFeaturesToTrack()的角点检测基础上将角点位置精确到亚像素级别


常见的亚像素级别精准定位方法有三类：
	1. 基于插值方法
	2. 基于几何矩寻找方法
	3. 拟合方法 - 比较常用

拟合方法中根据使用的公式不同可以分为
	1. 高斯曲面拟合与
	2. 多项式拟合等等。

以高斯拟合为例:

	窗口内的数据符合二维高斯分布
	Z = n / (2 * pi * 西格玛^2) * exp(-P^2/(2*西格玛^2))
	P = sqrt( (x-x0)^2 + (y-y0)^2)

	x,y   原来 整数点坐标
	x0,y0 亚像素补偿后的 坐标 需要求取

	ln(Z) = n0 + x0/(西格玛^2)*x +  y0/(西格玛^2)*y - 1/(2*西格玛^2) * x^2 - 1/(2*西格玛^2) * y^2
		n0 +            n1*x + n2*y +             n3*x^2 +              n3 * y^2
	   
	对窗口内的像素点 使用最小二乘拟合 得到上述 n0 n1 n2 n3
	  则 x0 = - n1/(2*n3)
	     y0 = - n2/(2*n3)



*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// 全局变量
Mat src, src_gray;

int maxCorners = 10;
int maxTrackbar = 50;//需要检测的角点数量

RNG rng(12345);//随机数 产生随机颜色 
char* source_window = "Image";
char* refineWindow = "refinement";  

/// 滑动条 回调函数 声明  亚像素级的角点检测
void goodFeaturesToTrack_Demo( int, void* );

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
  createTrackbar( "Max  corners:", source_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo);

  imshow( source_window, src );

  goodFeaturesToTrack_Demo( 0, 0 );

  waitKey(0);
  return(0);
}

// 亚像素级的角点检测
void goodFeaturesToTrack_Demo( int, void* )
{
  if( maxCorners < 1 ) { maxCorners = 1; }

  /// Parameters for Shi-Tomasi algorithm
  vector<Point2f> corners;//角点容器
  double qualityLevel = 0.01;
  double minDistance = 10;
  int blockSize = 3;//滑窗大小
  bool useHarrisDetector = false;
  double k = 0.04;// harris 角点阈值

  /// 复制源图像
  Mat copy, refineSrcCopy;
  copy = src.clone();
  refineSrcCopy = src.clone();
  /// 检测 Shi-Tomasi 角点
  goodFeaturesToTrack( src_gray,
                       corners,
                       maxCorners,//最多角点数量
                       qualityLevel,
                       minDistance,
                       Mat(),
                       blockSize,
                       useHarrisDetector,
                       k );


  /// 显示 角点
  cout<<"** Number of corners detected: "<<corners.size()<<endl;
  int r = 4;//圆半径 
  for( int i = 0; i < corners.size(); i++ )
     { 
	circle( copy, corners[i], r, Scalar(rng.uniform(0,255), 
					    rng.uniform(0,255),
                                            rng.uniform(0,255)), -1, 8, 0 ); 
     }
  /// 显示图
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, copy );

  /// 设置亚像素 检测补偿算法 参数
  Size winSize = Size( 5, 5 );//滑动窗大小
  Size zeroZone = Size( -1, -1 );// 用于避免自相关矩阵的奇异性
  // 角点精准化迭代过程的终止条件
  TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );// 最大迭代次数 40  和 误差变换 0.001

  /// 亚像素修正
  cornerSubPix( src_gray, // 输入图像
		corners,  // 输入角点的初始坐标以及精准化后的坐标用于输出
		winSize,  // 搜索窗口边长的一半  如果winSize=Size(5,5) 实际为 5*2+1 = 11 11*11的窗口
		zeroZone, // 搜索区域中间的dead region边长的一半，有时用于避免自相关矩阵的奇异性。如果值设为(-1,-1)则表示没有这个区域。
		criteria  // 角点精准化迭代过程的终止条件。也就是当迭代次数超过criteria.maxCount，
			  // 或者角点位置变化小于criteria.epsilon时，停止迭代过程。
			);

  /// 显示 修正后的角点坐标
  for( int i = 0; i < corners.size(); i++ )
     { 
        // 标示出角点  
        circle( refineSrcCopy, corners[i], r, Scalar(255,0,255), -1, 8, 0 );  
        // 输出角点坐标  
	cout<<" -- Refined Corner ["<<i<<"]  ("<<corners[i].x<<","<<corners[i].y<<")"<<endl; 
     }
  namedWindow( refineWindow, CV_WINDOW_AUTOSIZE );  
  imshow( refineWindow, refineSrcCopy );
}
