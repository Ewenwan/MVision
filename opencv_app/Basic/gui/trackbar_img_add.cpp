/*
为程序界面添加滑动条
图像 叠加
dst=α⋅src1+β⋅src2+ γ
α + β =1 
β = 1 - α 

*/



#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

/// 全局变量声明
const int alpha_slider_max = 100;
int alpha_slider;
double alpha, beta;

Mat img1, img2, dest;
//显示窗口名字
string window_name("blending混合");

// 滑动条响应回调函数
void on_trackbar_blend(int, void*){
alpha = (double)alpha_slider/alpha_slider_max ;
beta = 1 - alpha;

//线性混合 γ = 0
addWeighted(img1, alpha, img2, beta, 0.0, dest);

imshow(window_name, dest);
// 更新显示

}


int main( int argc, char** argv )
{
 //double alpha = 0.5; double beta; double input;

 img1 = imread("../../common/data/LinuxLogo.jpg");//读取
 img2 = imread("../../common/data/WindowsLogo.jpg");

 if( !img1.data ) { printf("Error loading img1 \n"); return -1; }
 if( !img2.data ) { printf("Error loading img2 \n"); return -1; }
 // 初始化为0
 alpha_slider = 0;
 // 创建窗体
 namedWindow(window_name, 1);

 // 创建滑动条
 char trackbarName[50];//滑动条左边显示的字体
 sprintf(trackbarName, "Alphe max %d", alpha_slider_max);
 // alphe_slider  change with the bar slide
 createTrackbar(trackbarName, window_name, &alpha_slider, alpha_slider_max, on_trackbar_blend);

 // 结果在回调函数中显示
 on_trackbar_blend( alpha_slider, 0 );

 // 按任意键退出
 waitKey(0);
 return 0;
}

