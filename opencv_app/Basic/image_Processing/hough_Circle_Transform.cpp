/*
霍夫 圆变换   在图像中检测圆. 
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
 string filename = argc >= 2 ? argv[1] : "../../common/data/apple.jpeg";

 Mat src = imread(filename, IMREAD_COLOR); // 按圆图片颜色 读取
 if(src.empty())
 {
     help();
     cout << "can not open " << filename << endl;
     return -1;
 }


    // 得到灰度图
    Mat src_gray = src.clone();
    if (src.channels() == 3)//如果原图是彩色图
    {
        cvtColor(src, src_gray, CV_BGR2GRAY);//转换到灰度图
    }
    //else
    //{
    //    src_gray = src.clone();
    //}

  // 高斯平滑降噪
  GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

// 执行霍夫圆变换
  vector<Vec3f> circles;//中性点（x，y）半价 r三个参数
  HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/10, 80, 50, 0, 0 );
// CV_HOUGH_GRADIENT: 指定检测方法. 现在OpenCV中只有霍夫梯度法
// dp = 1: 累加器图像的反比分辨率
// min_dist = src_gray.rows/8: 检测到圆心之间的最小距离
// param_1 = 200: Canny边缘函数的高阈值
// param_2 = 100: 圆心检测阈值.
// min_radius = 0: 能检测到的最小圆半径, 默认为0.
// max_radius = 0: 能检测到的最大圆半径, 默认为0

  for( size_t i = 0; i < circles.size(); i++ )
  {
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));//圆中心点
    int radius = cvRound(circles[i][2]);//半径 像素值单位
    // 画圆中心点 circle center
    circle( src, center, 3, Scalar(0,255,255), -1, 8);// green 绿色  粗细 线形
    // 画圆外圈   circle outline
    circle( src, center, radius, Scalar(0,0,255), 3, 8);// red 红色 bgr
   }

  
 namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
 imshow("Hough Circle Transform Demo", src);
 //imshow("detected circles", cdst);

 waitKey(0);// 等待用户按键结束程序

 return 0;
}


