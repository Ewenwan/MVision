/*LBP特征
级联分类器
*与Haar特征相比，LBP特征是整数特征，因此训练和检测过程都会比Haar特征快几倍。
LBP和Haar特征用于检测的准确率，是依赖训练过程中的训练数据的质量和训练参数。
训练一个与基于Haar特征同样准确度的LBP的分类器是可能的。

级联分类器的策略是，将若干个（ AdaBoost训练出来的强分类器 ） 由简单到复杂排列，
	希望经过训练使每个强分类器都有较高检测率，而误识率可以放低。

 */
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//函数声明
void detectAndDisplay( Mat frame );

//全局变量
String face_cascade_name = "../../common/data/cascade/lbpcascades/lbpcascade_frontalface.xml";
String eyes_cascade_name = "../../common/data/cascade/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";

// @主函数
int main( void )
{
	VideoCapture capture;
	Mat frame;

//==========【1】 加载级联分类器文件模型==============
	// 加载级联分类器
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

//==========【2】打开内置摄像头视频流==============
	capture.open( -1 );
	if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

	while ( capture.read(frame) )
	{
	if( frame.empty() )
	{
	    printf(" --(!) No captured frame -- Break!");
	    break;
	}

//=========【3】对当前帧使用分类器进行检测============
	detectAndDisplay( frame );

	//-- bail out if escape was pressed
	int c = waitKey(10);
	if( (char)c == 27 ) { break; }
	}
	return 0;
}

// @函数 detectAndDisplay 
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;//检测到的人脸 矩形区域 左下点坐标 长和宽
	Mat frame_gray;

	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );//转换成灰度图
	equalizeHist( frame_gray, frame_gray );//直方图均衡画

	//-- 多尺寸检测人脸
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(80, 80) );
	// image：当然是输入图像了，要求是8位无符号图像，即灰度图
	//objects：输出向量容器（保存检测到的物体矩阵）
	//scaleFactor：每张图像缩小的尽寸比例 1.1 即每次搜索窗口扩大10%
	//minNeighbors：每个候选矩阵应包含的像素领域
	//flags:表示此参数模型是否更新标志位； 0|CV_HAAR_SCALE_IMAGE   0|CASCADE_FIND_BIGGEST_OBJECT  CASCADE_DO_ROUGH_SEARCH
	//minSize ：表示最小的目标检测尺寸；
	//maxSize：表示最大的目标检测尺寸；
	for( size_t i = 0; i < faces.size(); i++ )
	{
	Mat faceROI = frame_gray( faces[i] );
	std::vector<Rect> eyes;

	//-- In each face, detect eyes
	eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
	if( eyes.size() == 2)
	{
	    // 画出椭圆区域
	    Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );//矩形中心点
	    ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );
	    //-- 在每张人脸上检测双眼
	    for( size_t j = 0; j < eyes.size(); j++ )
	    { //-- Draw the eyes // 中心点
		Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
		int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
		circle( frame, eye_center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );// 画圆
	    }
	}

	}
//====显示结果图像 显示人脸椭圆区域  人眼睛=====
	imshow( window_name, frame );
}


