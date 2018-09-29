/*
双目匹配　使用自己定义的类
./my_stereo 
 */
#include <opencv2/opencv.hpp> 
#include <string>
#include <iostream> 
#include "StereoMatch.h"//双目匹配类
#include "imageprocessor.h"//图像处理类
#include <pcl/visualization/cloud_viewer.h>//点云可视化

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    string config_filename("../date/config.yml"); // 图片　参数文件
    if( argc > 1)
     {
       config_filename = argv[1];//如果传递了文件 就更新
     }

    StereoMatch StereoM(640, 480, config_filename.c_str());
    StereoM.init(640, 480, config_filename.c_str());
    ImageProcessor iP(0.1);// 图像预处理类  像素数量阈值比率
    cv::Mat src_img, imgL, imgR, disp, disp8, disp32, dispImg;   
    cv::VideoCapture CapAll(1); //打开相机设备 
    if( !CapAll.isOpened() ) {printf("打开摄像头失败\r\n"); return -1;}
    //设置分辨率   1280*480  分成两张 640*480  × 2 左右相机
    CapAll.set(CV_CAP_PROP_FRAME_WIDTH,1280);  
    CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480);  
    // pcl::visualization::CloudViewer viewer("pcd　viewer");// 显示窗口的名字
    
    while(CapAll.read(src_img)) 
	{  
		imgL= src_img(cv::Range(0, 480), cv::Range(0, 640));   
		//imread(file,0) 灰度图  imread(file,1) 彩色图  默认彩色图
		imgR= src_img(cv::Range(0, 480), cv::Range(640, 1280));  
		//imgL = iP.unsharpMasking(imgL, "gauss", 3, 1.9, -1);
		//imgL = iP.unsharpMasking(imgL, "median", 5, 0.2, 0.8);
		//imgR = iP.unsharpMasking(imgR, "median", 5, 0.2, 0.8);
	 	StereoM.bmMatch(imgL, imgR, disp);// 速度最快
		//StereoM.sgbmMatch(imgL, imgR, disp);// 性能尚可
		//StereoM.hhMatch(imgL, imgR, disp);
		//StereoM.wayMatch(imgL, imgR, disp);//效果太差
                //StereoM.elasMatch(imgL, imgR, disp);// 速度太慢=!!! 计算的视差有问题====
		StereoM.getDisparityImage(disp, dispImg, true);
                disp.convertTo(disp8, CV_8U);
                // 计算出的视差都是CV_16S格式的，使用32位float格式可以得到真实的视差值，所以我们需要除以16
                disp.convertTo(disp32, CV_32F, 1.0/16); 
		disp32 = iP.unsharpMasking(disp32, "median", 5, 0, 1);
		namedWindow("左相机", 1);
		imshow("左相机", imgL);
		//namedWindow("视差", 1);
		//imshow("视差", disp8);
		namedWindow("视差彩色图", 1);
		imshow("视差彩色图", dispImg);
//                PointCloud::Ptr pointCloud_PCL2( new PointCloud ); 
//		PointCloud& my_color_pc = *pointCloud_PCL2;
//		StereoM.getPCL(disp, imgL, my_color_pc);
 //               StereoM.my_getpc(disp32, imgL, my_color_pc);
//		viewer.showCloud(pointCloud_PCL2);
		//printf("press any key to continue...");
		fflush(stdout);
		char c = waitKey(1);
		//printf("\n");
		//      } 
		if( c == 27 || c == 'q' || c == 'Q' )//按ESC/q/Q退出  
			break; 
	}

    return 0;
}
