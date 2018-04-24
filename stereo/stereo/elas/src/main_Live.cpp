#include "elas.h"
#include <opencv2/opencv.hpp>
using namespace cv;
int main()
{
	//Mat leftim=imread("left01.jpg");
	//Mat rightim=imread("right01.jpg");
  
        //Mat leftim=imread("view1.png");
	//Mat rightim=imread("view5.png");
	cv::Mat src_img,imgL, imageR;   
	cv::VideoCapture CapAll(1); //打开相机设备 
	if( !CapAll.isOpened() ) printf("打开摄像头失败\r\n");
	//设置分辨率   1280*480  分成两张 640*480  × 2 左右相机
	CapAll.set(CV_CAP_PROP_FRAME_WIDTH,1280);  
	CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	StereoELAS elas(0,128);// 最小视差  视差范围
	while(CapAll.read(src_img)) 
	{  
		imgL= src_img(cv::Range(0, 480), cv::Range(0, 640));   
//imread(file,0) 灰度图  imread(file,1) 彩色图  默认彩色图
		imageR= src_img(cv::Range(0, 480), cv::Range(640, 1280));    	

		Mat dest;

		// we can set various parameter
		//elas.elas.param.ipol_gap_width=;
		//elas.elas.param.speckle_size=getParameter("speckle_size");
		//elas.elas.param.speckle_sim_threshold=getParameter("speckle_sim");
		elas(imgL,imageR,dest,100);// 边界延拓

		Mat show;
		dest.convertTo(show,CV_8U,1.0/8);
		imshow("disp",show);
		//imwrite("disp3.jpg",show);
		waitKey();
        }
	return 0;
}
