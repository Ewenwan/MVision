/* 级联分类器 （CascadeClassifier） 
 AdaBoost强分类器串接
级联分类器是将若干个分类器进行连接，从而构成一种多项式级的强分类器。
从弱分类器到强分类器的级联（AdaBoost 集成学习  改变训练集）
级联分类器使用前要先进行训练，怎么训练？
用目标的特征值去训练，对于人脸来说，通常使用Haar特征进行训练。



【1】提出积分图(Integral image)的概念。在该论文中作者使用的是Haar-like特征，
	然后使用积分图能够非常迅速的计算不同尺度上的Haar-like特征。
【2】使用AdaBoost作为特征选择的方法选择少量的特征在使用AdaBoost构造强分类器。
【3】以级联的方式，从简单到 复杂 逐步 串联 强分类器，形成 级联分类器。

级联分类器。该分类器由若干个简单的AdaBoost强分类器串接得来。
假设AdaBoost分类器要实现99%的正确率，1%的误检率需要200维特征，
而实现具有99.9%正确率和50%的误检率的AdaBoost分类器仅需要10维特征，
那么通过级联，假设10级级联，最终得到的正确率和误检率分别为:
(99.9%)^10 = 99%
(0.5)^10   = 0.1

检测体系：是以现实中很大一副图片作为输入，然后对图片中进行多区域，多尺度的检测，
所谓多区域，是要对图片划分多块，对每个块进行检测，由于训练的时候一般图片都是20*20左右的小图片，
所以对于大的人脸，还需要进行多尺度的检测。多尺度检测一般有两种策略，一种是不改变搜索窗口的大小，
而不断缩放图片，这种方法需要对每个缩放后的图片进行区域特征值的运算，效率不高，而另一种方法，
是不断初始化搜索窗口size为训练时的图片大小，不断扩大搜索窗口进行搜索。
在区域放大的过程中会出现同一个人脸被多次检测，这需要进行区域的合并。
无论哪一种搜索方法，都会为输入图片输出大量的子窗口图像，
这些子窗口图像经过筛选式级联分类器会不断地被每个节点筛选，抛弃或通过。


级联分类器的策略是，将若干个强分类器由简单到复杂排列，
希望经过训练使每个强分类器都有较高检测率，而误识率可以放低。

AdaBoost训练出来的强分类器一般具有较小的误识率，但检测率并不很高，
一般情况下，高检测率会导致高误识率，这是强分类阈值的划分导致的，
要提高强分类器的检测率既要降低阈值，要降低强分类器的误识率就要提高阈值，
这是个矛盾的事情。据参考论文的实验结果，
增加分类器个数可以在提高强分类器检测率的同时降低误识率，
所以级联分类器在训练时要考虑如下平衡，一是弱分类器的个数和计算时间的平衡，
二是强分类器检测率和误识率之间的平衡。



*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//函数声明
void detectAndDisplay( Mat frame );

//全局变量
string face_cascade_name = "../../common/data/cascade/haarcascades/haarcascade_frontalface_alt.xml";
string eyes_cascade_name = "../../common/data/cascade/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
// 级联分类器 类
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

string window_name = "Capture - Face detection";
//RNG rng(12345);

// @主函数
int main( int argc, const char** argv )
{
	//CvCapture* capture;
	VideoCapture capture;
	Mat frame;

//==========【1】 加载级联分类器文件模型==============
	// 加载级联分类器
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

//==========【2】打开内置摄像头视频流==============
	capture.open( -1 );
        if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
	while( capture.read(frame) )
	{
//=========【3】对当前帧使用分类器进行检测============
		if( !frame.empty() )
		{ detectAndDisplay( frame ); }
		else
		{ printf(" --(!) No captured frame -- Break!"); break; }

		int c = waitKey(10);
		if( (char)c == 'c' ) { break; }
	}
	
	return 0;
}

// @函数 detectAndDisplay 
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;//检测到的人脸 矩形区域 左下点坐标 长和宽
	Mat frame_gray;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );//转换成灰度图
	equalizeHist( frame_gray, frame_gray );//直方图均衡画

	//-- 多尺寸检测人脸
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	// image：当然是输入图像了，要求是8位无符号图像，即灰度图
	//objects：输出向量容器（保存检测到的物体矩阵）
	//scaleFactor：每张图像缩小的尽寸比例 1.1 即每次搜索窗口扩大10%
	//minNeighbors：每个候选矩阵应包含的像素领域
	//flags:表示此参数模型是否更新标志位； 0|CV_HAAR_SCALE_IMAGE   0|CASCADE_FIND_BIGGEST_OBJECT  CASCADE_DO_ROUGH_SEARCH
	//minSize ：表示最小的目标检测尺寸；
	//maxSize：表示最大的目标检测尺寸；
	//
	for( int i = 0; i < faces.size(); i++ )//画出椭圆区域
	{//矩形中心点
	   Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
	   ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

	   Mat faceROI = frame_gray( faces[i] );//对应区域的像素图片
	   std::vector<Rect> eyes;//眼睛

	   //-- 在每张人脸上检测双眼
	   eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 
					0 
					//|CASCADE_FIND_BIGGEST_OBJECT
					//|CASCADE_DO_ROUGH_SEARCH
					|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	   for( int j = 0; j < eyes.size(); j++ )
	   {    // 中心点
		Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5,
		              faces[i].y + eyes[j].y + eyes[j].height*0.5 );
		int radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
		circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );// 画圆
	}
	}
	//-- 显示结果图像 显示人脸椭圆区域  人眼睛
	imshow( window_name, frame );
}

