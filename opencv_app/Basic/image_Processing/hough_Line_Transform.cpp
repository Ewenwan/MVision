/*
霍夫线变换  检测图像中的直线 先candy边缘检测 在找直线
使用OpenCV的以下函数 HoughLines 和 HoughLinesP 来检测图像中的直线.

霍夫线变换
    霍夫线变换是一种用来寻找直线的方法.
    是用霍夫线变换之前, 首先要对图像进行边缘检测的处理，
    也即霍夫线变换的直接输入只能是边缘二值图像.

众所周知, 一条直线在图像二维空间可由两个变量表示. 例如:

    在 笛卡尔坐标系: 可由参数: (m,b) 斜率和截距表示.  y = m*x + b
    在 极坐标系: 可由参数: (r,\theta) 极径和极角表示  r = x * cos(theta) + y*sin(theta)

   一般来说, 一条直线能够通过在平面 theta - r 寻找交于一点的曲线数量来 检测. 
   越多曲线交于一点也就意味着这个交点表示的直线由更多的点组成. 
   一般来说我们可以通过设置直线上点的 阈值 来定义多少条曲线交于一点我们才认为 检测 到了一条直线.

   这就是霍夫线变换要做的. 它追踪图像中每个点对应曲线间的交点. 
   如果交于一点的曲线的数量超过了 阈值, 
   那么可以认为这个交点所代表的参数对 (theta, r_{theta}) 在原图像中为一条直线.



标准霍夫线变换和统计概率霍夫线变换
OpenCV实现了以下两种霍夫线变换:
    【1】标准霍夫线变换
	它能给我们提供一组参数对 (\theta, r_{\theta}) 的集合来表示检测到的直线
        在OpenCV 中通过函数 HoughLines 来实现

    【2】统计概率霍夫线变换

        这是执行起来效率更高的霍夫线变换. 
        它输出检测到的直线的端点 (x_{0}, y_{0}, x_{1}, y_{1})
        在OpenCV 中它通过函数 HoughLinesP 来实现
*/


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void help()
{
 cout << "\nThis program demonstrates line finding with the Hough transform.\n"
         "Usage:\n"
         "./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}

int main(int argc, char** argv)
{
 const char* filename = argc >= 2 ? argv[1] : "../../common/data/77.jpeg";

 Mat src = imread(filename, 0);
 if(src.empty())
 {
     help();
     cout << "can not open " << filename << endl;
     return -1;
 }

 Mat dst, cdst;
// 检测边缘
 Canny(src, dst, 50, 200, 3);//低阈值 高阈值 核尺寸

 cvtColor(dst, cdst, CV_GRAY2BGR);//灰度图

//【1】标准霍夫线变换
 #if 0
  vector<Vec2f> lines;//得到 直线的参数 r theta
  HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

  for( size_t i = 0; i < lines.size(); i++ )
  {
     float rho = lines[i][0], theta = lines[i][1];
     Point pt1, pt2;
     double a = cos(theta), b = sin(theta);
     double x0 = a*rho, y0 = b*rho;
     pt1.x = cvRound(x0 + 1000*(-b));
     pt1.y = cvRound(y0 + 1000*(a));
     pt2.x = cvRound(x0 - 1000*(-b));
     pt2.y = cvRound(y0 - 1000*(a));
     line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
  }
 #else
// 【2】统计概率霍夫线变换
  vector<Vec4i> lines;//直线首尾点
  HoughLinesP(dst, lines, 1, CV_PI/180, 100, 50, 10 );
// 以像素值为单位的分辨率. 我们使用 1 像素.
// theta: 参数极角  theta 以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
// threshold: 要”检测” 一条直线所需最少的的曲线交点 50
// minLineLength = 0,  最小线长
// maxLineGap = 0 , 最大线间隔  maxLineGap: 能被认为在一条直线上的亮点的最大距离.
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
  }
 #endif

 imshow("source", src);
 imshow("detected lines", cdst);

 waitKey();

 return 0;
}


