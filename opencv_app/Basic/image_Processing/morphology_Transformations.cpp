/*
使用OpenCV函数 morphologyEx 进行形态学操作：
    开运算 (Opening)
    闭运算 (Closing)
    形态梯度 (Morphological Gradient)
    顶帽 (Top Hat)
    黑帽(Black Hat)

【1】开运算 (Opening)   去除 小型 白洞
    开运算是通过先对图像腐蚀再膨胀实现的。
    dst = open( src, element) = dilate( erode( src, element ) )
    能够排除小团块物体(假设物体较背景明亮)
    请看下面，左图是原图像，右图是采用开运算转换之后的结果图。 
    观察发现字母拐弯处的白色空间消失。


【2】闭运算(Closing)
    闭运算是通过先对图像膨胀再腐蚀实现的。
    dst = close( src, element ) = erode( dilate( src, element ) )
    能够排除小型黑洞(黑色区域)

【3】形态梯度(Morphological Gradient)
    膨胀图与腐蚀图之差
    dst = morph_{grad}( src, element ) = dilate( src, element ) - erode( src, element )
    能够保留物体的边缘轮廓.

【4】顶帽(Top Hat)
    原图像与开运算结果图之差
    dst = tophat( src, element ) = src - open( src, element )


【5】黑帽(Black Hat)
    闭运算结果图与原图像之差
    黑变白  白变黑
    dst = blackhat( src, element ) = close( src, element ) - src



*/
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

// 全局变量
Mat src, dst;
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;

string window_name("Morphology Transformations 形态学变换 Demo");

// 回调函数声明 
void Morphology_Operations( int, void* );


/* @function main */
int main( int argc, char** argv )
{

   namedWindow( window_name, WINDOW_AUTOSIZE );//新建窗口显示
   src = imread( "../../common/data/77.jpeg", 1 );
   if(src.empty()) {
       cout << "can't load image " << endl;
       return -1;
   }

// 创建显示窗口
 namedWindow( window_name, WINDOW_AUTOSIZE );

// 创建选择具体操作的 滑动条trackbar  动态改变参数 
 createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations );//回调函数

// 创建选择内核形状的  矩形  交叉形  椭圆形
 createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
         &morph_elem, max_elem,// 参数 morph_elem 上限 max_elem
         Morphology_Operations );

// 核大小
 createTrackbar( "Kernel size:\n 2n +1", window_name,
         &morph_size, max_kernel_size,
         Morphology_Operations );

// 默认 开运算  矩阵 1大小
 Morphology_Operations( 0, 0 );
 waitKey(0);
 return 0;
 }

 /*
  * 形态学操作  Morphology_Operations
  */
void Morphology_Operations( int, void* )
{
  // MORPH_X 取值范围是: 2,3,4,5 和 6 
  int operation = morph_operator + 2;
  // 参数 形状  大小  锚点
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  // 运行指定形态学操作
  morphologyEx( src, dst, operation, element );
  imshow( window_name, dst );
  }






