/*
图像 叠加
dst=α⋅src1+β⋅src2+ γ
α + β =1 


*/

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
int main( int argc, char** argv )
{
 double alpha = 0.5; double beta; double input;
 Mat src1, src2, dst;
 std::cout<<" Simple Linear Blender "<<std::endl;
 std::cout<<"-----------------------"<<std::endl;
 std::cout<<"* Enter alpha [0-1]: ";
 std::cin>>input;//字符串转换成 double

 if( input >= 0.0 && input <= 1.0 )
   { alpha = input; }

 src1 = imread("../data/LinuxLogo.jpg");//读取
 src2 = imread("../data/WindowsLogo.jpg");

 if( !src1.data ) { printf("Error loading src1 \n"); return -1; }
 if( !src2.data ) { printf("Error loading src2 \n"); return -1; }

 namedWindow("Linear Blend", 1);
 beta = ( 1.0 - alpha );
 addWeighted( src1, alpha, src2, beta, 0.0, dst);//图像叠加
 imshow( "Linear Blend", dst );

 waitKey(0);
 return 0;
}

