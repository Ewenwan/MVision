/*
通过自定义 R的计算方法和自适应阈值 来定制化检测角点

计算 M矩阵
计算判断矩阵 R

设置自适应阈值

阈值大小为 判断矩阵 最小值和最大值之间 百分比
阈值为 最小值 + （最大值-最小值）× 百分比
百分比 = myHarris_qualityLevel/max_qualityLevel

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

*/
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;

// 全局变量
Mat src, src_gray;
Mat myHarris_dst; Mat myHarris_copy; Mat Mc;// Harris角点角点相关 判断矩阵R
Mat myShiTomasi_dst; Mat myShiTomasi_copy;  // Shi-Tomasi 角点检测算法 

int myShiTomasi_qualityLevel = 50;// Shi-Tomasi 角点检测算法 阈值
int myHarris_qualityLevel = 50;   // Harris角点角点检测算法 阈值
int max_qualityLevel = 100;       // 最大阈值  百分比  myHarris_qualityLevel/max_qualityLevel
// 阈值为 最小值 + （最大值-最小值）× 百分比

double myHarris_minVal; double myHarris_maxVal;       // Harris角点 判断矩阵的最小最大值
double myShiTomasi_minVal; double myShiTomasi_maxVal;// Shi-Tomasi 角点 判断矩阵的最小最大值

RNG rng(12345);//随机数  产生 随机颜色

const char* myHarris_window = "My Harris corner detector";
const char* myShiTomasi_window = "My Shi Tomasi corner detector";

//函数声明
void myShiTomasi_function( int, void* );
void myHarris_function( int, void* );


int main( int argc , char** argv )
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
  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  int blockSize = 3; // 滑动窗口大小
  int apertureSize = 3;// Sobel算子的大小(默认值为3) 用来计算 梯度 Ix^2   Ix*Iy Iy^2 
//============== Harris角点==========================
  myHarris_dst = Mat::zeros( src_gray.size(), CV_32FC(6) );//得到的 矩阵M的特征值
  Mc = Mat::zeros( src_gray.size(), CV_32FC1 );//定制化的 判断矩阵 R
  // 计算 矩阵M的特征值
  cornerEigenValsAndVecs( src_gray, myHarris_dst, blockSize, apertureSize, BORDER_DEFAULT );
  // 计算判断矩阵 R
  for( int j = 0; j < src_gray.rows; j++ )
     { for( int i = 0; i < src_gray.cols; i++ )
          {
            float lambda_1 = myHarris_dst.at<Vec6f>(j, i)[0];
            float lambda_2 = myHarris_dst.at<Vec6f>(j, i)[1];
            Mc.at<float>(j,i) = lambda_1*lambda_2 - 0.04f*pow( ( lambda_1 + lambda_2 ), 2 );
          }
     } // 判断矩阵 R 的最低值和最高值
  minMaxLoc( Mc, &myHarris_minVal, &myHarris_maxVal, 0, 0, Mat() );
  //创建一个窗口 和 滑动条
  namedWindow( myHarris_window, WINDOW_AUTOSIZE );
  createTrackbar( " Quality Level:", myHarris_window, &myHarris_qualityLevel, max_qualityLevel, myHarris_function );
  myHarris_function( 0, 0 );

//===================Shi-Tomasi 角点检测算法===============================
  myShiTomasi_dst = Mat::zeros( src_gray.size(), CV_32FC1 );// Harris角点角点检测算法 判断矩阵  min(lambda_1 , lambda_2)
  cornerMinEigenVal( src_gray, myShiTomasi_dst, blockSize, apertureSize, BORDER_DEFAULT );
  minMaxLoc( myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal, 0, 0, Mat() );
  //创建一个窗口 和 滑动条
  namedWindow( myShiTomasi_window, WINDOW_AUTOSIZE );
  createTrackbar( " Quality Level:", myShiTomasi_window, &myShiTomasi_qualityLevel, max_qualityLevel, myShiTomasi_function );
  myShiTomasi_function( 0, 0 );
  waitKey(0);
  return(0);
}
// Shi-Tomasi 角点检测算法 滑动条回调函数
void myShiTomasi_function( int, void* )
{

  if( myShiTomasi_qualityLevel < 1 ) 
	{ 
	  myShiTomasi_qualityLevel = 1; 
	}
  // 自适应阈值 
  // 阈值为 最小值 + （最大值-最小值）× 百分比
  // 百分比 = yShiTomasi_qualityLevel/max_qualityLevel
  float thresh = myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal ) * myShiTomasi_qualityLevel/max_qualityLevel;
  myShiTomasi_copy = src.clone();
  for( int j = 0; j < src_gray.rows; j++ )
     { 
	for( int i = 0; i < src_gray.cols; i++ )
          {
            if( myShiTomasi_dst.at<float>(j,i) > thresh)
            { // 在角点处画圆圈
	    circle( myShiTomasi_copy, Point(i,j), 4, Scalar( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255) ), -1, 8, 0 ); 
            }
         }
     }
  imshow( myShiTomasi_window, myShiTomasi_copy );
}
 

// Harris角点 滑动条回调函数
void myHarris_function( int, void* )
{

  if( myHarris_qualityLevel < 1 ) 
	{ 
	myHarris_qualityLevel = 1; 
	}

  // 自适应阈值 
  // 阈值为 最小值 + （最大值-最小值）× 百分比
  // 百分比 = myHarris_qualityLevel/max_qualityLevel
  float thresh = myHarris_minVal + ( myHarris_maxVal - myHarris_minVal ) * myHarris_qualityLevel/ max_qualityLevel;
  myHarris_copy = src.clone();
  for( int j = 0; j < src_gray.rows; j++ )
     { for( int i = 0; i < src_gray.cols; i++ )
          {
            if( Mc.at<float>(j,i) >  thresh )
             { // 在角点处画圆圈
	     circle( myHarris_copy, Point(i,j), 4, Scalar( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255) ), -1, 8, 0 );
 	     }
          }
     }
  imshow( myHarris_window, myHarris_copy );
}
