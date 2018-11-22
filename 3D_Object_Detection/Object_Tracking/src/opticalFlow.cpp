// API calcOpticalFlowFarneback() comes from OpenCV, and this 
// 2D dense optical flow algorithm from the following paper: 
// Gunnar Farneback. "Two-Frame Motion Estimation Based on Polynomial Expansion". 
// And the OpenCV source code locate in ..\opencv2.4.3\modules\video\src\optflowgf.cpp 

// https://blog.csdn.net/ironyoung/article/details/60884929
// https://blog.csdn.net/yzhang6_10/article/details/51225545

#include <iostream> 
#include "opencv2/opencv.hpp" 
using namespace cv; 
using namespace std; 
#define UNKNOWN_FLOW_THRESH 1e9 // 光流最大值 1000000000

// 孟塞尔颜色系统===========
// Color encoding of flow vectors from: 
// http://members.shaw.ca/quadibloc/other/colint.htm 
// This code is modified from: 
// http://vision.middlebury.edu/flow/data/
void makecolorwheel(vector<Scalar> &colorwheel) 
{ 
// 红(R)、红黄(YR)、黄(Y)、黄绿(GY)、绿(G)、绿蓝(BG)、蓝(B)、蓝紫(PB)、紫(P)、紫红(RP)。
	int RY = 15; 
	int YG = 6; 
	int GC = 4; 
	int CB = 11; 
	int BM = 13; 
	int MR = 6; 
	int i; 
	for (i = 0; i < RY; i++) 
		colorwheel.push_back(Scalar(255, 255*i/RY, 0)); 
	for (i = 0; i < YG; i++) 
		colorwheel.push_back(Scalar(255-255*i/YG, 255, 0));
	 for (i = 0; i < GC; i++) 
		colorwheel.push_back(Scalar(0, 255, 255*i/GC)); 
	for (i = 0; i < CB; i++) 
		colorwheel.push_back(Scalar(0, 255-255*i/CB, 255)); 
	for (i = 0; i < BM; i++) 
		colorwheel.push_back(Scalar(255*i/BM, 0, 255)); 
	for (i = 0; i < MR; i++) 
		colorwheel.push_back(Scalar(255, 0, 255-255*i/MR)); 
} 

// 输入的flow 没一点包含两个值，水平光流 和 垂直光流
// 输出的 color, 包含3个值，光流转换成的  r,g,b 
void motionToColor(Mat flow, Mat &color) 
{ 
// 彩色光流图3通道============
if (color.empty()) 
	color.create(flow.rows, flow.cols, CV_8UC3); 

static vector<Scalar> colorwheel; //Scalar r,g,b 
if (colorwheel.empty()) 
	makecolorwheel(colorwheel); 

// determine motion range 
float maxrad = -1; // 综合光流最大值
// 找到最大的光流值，来归一化 水平和垂直光流===========================
// Find max flow to normalize fx and fy
for (int i= 0; i < flow.rows; ++i) 
{ 
	for (int j = 0; j < flow.cols; ++j) 
        { 
          Vec2f flow_at_point = flow.at<Vec2f>(i, j);// 光流值
          float fx = flow_at_point[0]; // 水平光流
          float fy = flow_at_point[1]; // 垂直光流
          // 值过大 就不符合
          if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH)) 
               continue; 
          // 计算综合光流，两直角边得到 斜边
          float rad = sqrt(fx * fx + fy * fy); 
          maxrad = maxrad > rad ? maxrad : rad;// 保留最大 综合光流
        } 
} 

 cout << "max flow:  " << maxrad << endl; // 打印一下 最大 综合光流值

// 这个flow颜色可视化分成这么几步： 
// 1） 对flow归一化后，算出它的极坐标 （angle, radius） 
//2） 将angle 映射到色调（hue）， 将radius 映射到色度(saturation)。 
// 这里共分了55个色调
for (int i= 0; i < flow.rows; ++i) // 行
{ 
	for (int j = 0; j < flow.cols; ++j) // 列
        { 
          uchar *data = color.data + color.step[0] * i + color.step[1] * j; // rgb图
          Vec2f flow_at_point = flow.at<Vec2f>(i, j);



      float tep = sqrt(flow_at_point[0] * flow_at_point[0] + flow_at_point[1] * flow_at_point[1]); 
         if(tep < 4.0) // 剔除过小的 光流值
          { 
            data[0] = data[1] = data[2] = 0;
                 continue; // 光流太大，假
          } 

          // 使用最大 综合光流 来归一化 水平和垂直光流 ======
          float fx = flow_at_point[0] / maxrad; 
          float fy = flow_at_point[1] / maxrad; 
// 这里有问题，fx，fy已经归一化了
          if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH)) 
          { 
            data[0] = data[1] = data[2] = 0;
                 continue; // 光流太大，假
          } 
          float rad = sqrt(fx * fx + fy * fy);   // 综合光流 赋值
          float angle = atan2(-fy, -fx) / CV_PI; // 综合光流 方向

          float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);// 角度选 颜色轮子
          int k0 = (int)fk; 

          int k1 = (k0 + 1) % colorwheel.size(); 
          float f = fk - k0; 
          //f = 0; // uncomment to see original color wheel
          for (int b = 0; b < 3; b++) 
          { 
            float col0 = colorwheel[k0][b] / 255.0; 
            float col1 = colorwheel[k1][b] / 255.0; 
            float col = (1 - f) * col0 + f * col1; 
            if (rad <= 1) 
                 col = 1 - rad * (1 - col); // increase saturation with radius
            else 
                 col *= .75; // out of range 
            data[2 - b] = (int)(255.0 * col); 
          } 
        } 
   } 
} 

int main(int, char**) 
{ 
	VideoCapture cap; cap.open(0); 
	//cap.open("test_02.wmv"); 
	if( !cap.isOpened() ) 
		return -1; 
	Mat prevgray, gray, flow, flow2, cflow, frame, frameSrc; 

	//namedWindow("flow", 1); 

	Mat motion2color,motion2color2 ;
        //cap >> frameSrc;
        
        // 动/静mask=====
        Mat mask;

	for(;;) 
	{
		 double t = (double)cvGetTickCount(); 
		 cap >> frameSrc;
                 mask = cv::Mat::zeros(frameSrc.rows,frameSrc.cols,CV_8U);// 默认0;
// 下采样一下 加快速度====================== 速度 ×3 ==================
	         pyrDown(frameSrc, frame, Size(frameSrc.cols / 2, frameSrc.rows / 2));
// ====================================================================================

		 cvtColor(frame, gray, CV_BGR2GRAY); // 转成灰度
		 //imshow("original", frame); 
		 if( prevgray.data ) 
		 { 

// CalcOpticalFlowFarneback()函数是利用用Gunnar Farneback的算法,
// 计算全局性的稠密光流算法（即图像上所有像素点的光流都计算出来），
// 由于要计算图像上所有点的光流，故计算耗时，速度慢。
// 参数说明如下： 
// _prev0：输入前一帧图像 
// _next0：输入后一帧图像 
// _flow0：输出的光流 
// pyr_scale：金字塔上下两层之间的尺度关系 
// levels：金字塔层数 
// winsize：均值窗口大小，越大越能denoise并且能够检测快速移动目标，但会引起模糊运动区域 
// iterations：迭代次数
 // poly_n：像素领域大小，一般为5，7等 
// poly_sigma：高斯标注差，一般为1-1.5 
// flags：计算方法。主要包括 OPTFLOW_USE_INITIAL_FLOW 和 OPTFLOW_FARNEBACK_GAUSSIAN 

		    calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0); 
		   // motionToColor(flow, motion2color); // 运动图转换到 色彩图

// 上采样 ======
         // pyrUp(motion2color, motion2color2, Size(motion2color.cols * 2, motion2color.rows * 2));
	  // imshow("flow", motion2color2); 

          pyrUp(flow, flow2, Size(flow.cols * 2, flow.rows * 2));

            /*
              // 显示光流方线段=====================================
                 for(int y=0; y<frame.rows; y+= 5) // 每隔5行画线
                 { 
                   for(int x=0; x<frame.cols; x+= 5)
                    { 
                      const Point2f xy = flow.at<Point2f>(y, x);
                      const Point2f flowatxy = xy*10;// 光流值放大10倍

		      float tep = sqrt(xy.x * xy.x + xy.y * xy.y); 
		      if(tep < 4.0) // 剔除过小的 光流值
		          continue; // 光流太大，假
                      line(frame, Point(x,y), 
                           Point(cvRound(x+flowatxy.x), cvRound(y+flowatxy.y)), 
                           Scalar(255, 0, 0));// 起点到 终点 画线
                      circle(frame, Point(x,y), 1, Scalar(0,0,0), -1); // 起点
                    } 
                 }
                 imshow("original", frame); 
              */

              // 显示光流方线段=====================================
                 for(int y=0; y<frameSrc.rows; y+= 5) // 每隔5行画线
                 { 
                   for(int x=0; x<frameSrc.cols; x+= 5)
                    { 
                      const Point2f xy = flow2.at<Point2f>(y, x);
		      float tep = sqrt(xy.x * xy.x + xy.y * xy.y); 
		      if(tep < 4.0) // 剔除过小的 光流值
		          continue; // 光流太大，假

                      mask.at<unsigned char>(y,x) = 255;

                      const Point2f flowatxy = xy*10;// 光流值放大10倍
                      /*line(frameSrc, Point(x,y), 
                           Point(cvRound(x+flowatxy.x), cvRound(y+flowatxy.y)), 
                           Scalar(255, 0, 0));// 起点到 终点 画线
                      circle(frameSrc, Point(x,y), 1, Scalar(0,0,0), -1); // 起点
                      */
                    } 
                 }
                 //imshow("original", frameSrc); 
		// 膨胀，选大的
		int dilation_size = 10;// 膨胀核大小===
		cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
				                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
				                       cv::Point( dilation_size, dilation_size ) );
		cv::dilate(mask, mask, kernel);// 膨胀，15×15核内 选最大的
                cv::dilate(mask, mask, kernel);// 膨胀，15×15核内 选最大的
                cv::dilate(mask, mask, kernel);// 膨胀，15×15核内 选最大的
                cv::erode(mask, mask, kernel);// 腐蚀
                imshow("mask", mask); 

		 } 


		 if(waitKey(10)>=0) break; 
		 std::swap(prevgray, gray); // 更新上一帧灰度图
		 t = (double)cvGetTickCount() - t; 
		 cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
	}
	return 0; 
}





