/*
图像进行平滑，也可以叫做模糊。平滑图像的主要目的是减少噪声，
这样采用平滑图像来降低噪声是是非常常见的预处理方法。
    1.归一化滤波平滑-Homogeneous Smoothing
    2.高斯滤波平滑-Gaussian Smoothing
    3.中值滤波平滑-Median Smoothing
    4.双边滤波平滑-Bilateral Smoothing

平滑是通过滑动窗口（内核或过滤器）扫描整个图像窗口，
计算每个像素的基于核的值的和重叠的原始图像的像素的值的值来完成
。这个过程在数学上称为具有一些内核卷积的图像。
上述4种不同平滑方法唯一的区别就是内核。

https://blog.csdn.net/tealex/article/details/51553787

【1】 5 x 5的核用来平滑（模糊）下面图片， 归一化块滤波器"Normalized box filter".
内核：
1/25 [1 1 1 1 1
      1 1 1 1 1
      1 1 1 1 1
      1 1 1 1 1
      1 1 1 1 1]
 简单的滤波器， 输出像素值是核窗口内像素值的 均值 ( 所有像素加权系数相等)
选择核的大小是很关键，如果选择太大，比较细微的图像特征可能被平滑掉，图像看起来很模糊。
如果核选择太小，就可能无法完全删除噪声。
平滑图像 -可以看到随着平滑核增大，图像逐渐变得模糊


【2】而高斯平滑（模糊）采用5x5的内核是如下。 二维高斯（中间权重大，越靠边权重低）
这个内核被称为“高斯核” "Gaussian kernel"


1/273 [1   4   7   4   1
       4   16  26  16  4
       7   26  41  26  7
       4   16  26  16  4
       1   4   7   4   1]
"高斯平滑" 也叫 "高斯模糊" 或 "高斯过滤".是比较常用的一个平滑方法。
是比较常见的平滑方法。也用于去除图像噪声。高斯核滑动扫描图像来对图像进行平滑。
内核的大小和在X和Y方向上的高斯分布的标准偏差应慎重选择。

【3】中值滤波  保留窗口内像素的中值  Median Smoothing
“平均平滑处理”也被称为“中间模糊处理”或“值滤波”。
这也是一种常见的平滑技术。输入图像进行卷积用中值的内核。
中值滤波是广泛用于边缘检测算法，因为在某些情况下，它保留边缘的同时去除噪声。

【4】
双边平滑 Bilateral Smoothing

“双边平滑处理”也被称为“双边模糊处理”或“双边滤波”。
这是最先进的过滤器，以平滑图像和减少噪音。
同时去除噪声上述所有过滤器会抚平的边缘。
但该过滤器能够减少图像的噪声，同时保留边缘。
这种类型的过滤器的缺点是，它需要较长的时间来处理。



平滑核（滤波器）比较重要的几个方面：
【1】平滑核的行像素数及列像素数必须是奇数(e.g. - 3x3, 11x5, 7x7, etc)；
【2】平滑核的size越大，计算时间越长。


*/
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;

using namespace cv;

int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;
Mat src; Mat dst;//源图像  变换后的图像

char window_name[] = "Filter Demo 1";//窗口名字

int display_caption( char* caption );

int display_dst( int delay );

/*
 * function main
 */
 int main( int argc, char** argv )
 {

   namedWindow( window_name, WINDOW_AUTOSIZE );//新建窗口显示
   src = imread( "../../common/data/77.jpeg", 1 );
   if(src.empty()) {
       cout << "can't load image " << endl;
       return -1;
   }
//显示原图像
   if( display_caption( "Original Image" ) != 0 ) {
     return 0; 
   }

   dst = src.clone();

   if( display_dst( DELAY_CAPTION ) != 0 ) { 
      return 0; 
   }
// 归一化块滤波器"Normalized box filter".
   if( display_caption( "Homogeneous Blur" ) != 0 ) { return 0; }
   for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )//注意 i = i + 2 加二 为奇数
        { 
// ksize - 核大小 anchor- 点(-1,-1)的值意味着anchor是核中心的值。如果愿意可以自己定义自己的点。
	 blur( src, dst, Size( i, i ), Point(-1,-1) );
         if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } 
        }

// "高斯平滑" 也叫 "高斯模糊" 或 "高斯过滤".是比较常用的一个平滑方法。
    if( display_caption( "Gaussian Blur" ) != 0 ) { return 0; }
    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
        { 
          GaussianBlur( src, dst, Size( i, i ), 0, 0 );
// 后面两个参数  在X方向的标准偏差。如果使用0，它会自动从内核尺寸计算
// 在Y方向的标准偏差。如果使用0，它会取和sigmaX一样
          if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } 
        }

// 中值滤波
 if( display_caption( "Median Blur" ) != 0 ) { return 0; }
 for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { 
	   medianBlur ( src, dst, i );// 滤波器的大小  单边
           if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } 
	 }

// 双边滤波
 if( display_caption( "Bilateral Blur" ) != 0 ) { return 0; }
 for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { 
// d - 像素领域直径 sigmaColor - 在颜色空间的sigma sigmaSpace - 在坐标空间的sigma
	   bilateralFilter ( src, dst, i, i*2, i/2 );
           if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } 
	 }

 display_caption( "End: Press a key!" );

 waitKey(0);//等待按键点击后 退出

 return 0;
 }

//显示 文字
 int display_caption( char* caption )
 {
   dst = Mat::zeros( src.size(), src.type() );
   putText( dst, caption,
            Point( src.cols/4, src.rows/2),//显示的点
            FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
//               字体             缩放    颜色
   imshow( window_name, dst );
   int c = waitKey( DELAY_CAPTION );
   if( c >= 0 ) { return -1; }
   return 0;
  }

//显示目标图像
  int display_dst( int delay )
  {
    imshow( window_name, dst );
    int c = waitKey ( delay );
    if( c >= 0 ) { return -1; }
    return 0;
  }


