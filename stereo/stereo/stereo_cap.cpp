/*
双目相机采集
双目相机类型：
双设备, video1 video2
单设备，拼接 1280*960 + 1280*960 >>> 2560*960 
*/
#include <iostream>  
#include <opencv2/opencv.hpp>  

using namespace std;  
using namespace cv;  
    
int main()  
{  

cv::VideoCapture CapAll(1);  
//VideoCapture capture;//捕获相机对象

//cv::VideoCapture capl(1);  
//v::VideoCapture capr(2);  
 if( !CapAll.isOpened() )//在线 矫正 
        printf("打开摄像头失败\r\n");
    
CapAll.set(CV_CAP_PROP_FRAME_WIDTH,1280);  
CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480);  
 
int i = 1;  
cv::Mat src_img;
cv::Mat src_imgl;  
cv::Mat src_imgr;  

char filename_l[15];  
char filename_r[15];  
//CapAll.read(src_img);

// Size  imageSize=src_img.size();
//printf( "%2d %2d", imageSize.width,imageSize.height);//图像大小);
//while(capl.read(src_imgl) && capr.read(src_imgr)) 
while(CapAll.read(src_img)) 
	{  
	//相机图像分离源码  
	    //双目相机图像2560*960    1280*480   
	    //单个相机图像1280*960     640*480
	    cv::imshow("src_img", src_img);  
	    

	    src_imgl = src_img(cv::Range(0, 480), cv::Range(0, 640));  
            src_imgr = src_img(cv::Range(0, 480), cv::Range(640, 1280));  
	    cv::imshow("src_imgl", src_imgl);  
	    cv::imshow("src_imgr", src_imgr);  

	    char c = cv::waitKey(1);  
	    if(c==' ') //按空格采集图像  
	    {  
	        sprintf(filename_l, "%s%02d%s","left", i,".jpg");  
		imwrite(filename_l, src_imgl);  
	        sprintf(filename_r, "%s%02d%s","right", i++,".jpg");  
		imwrite(filename_r, src_imgr); 
		//++i;
		//ReleaseImage(& src_imgl; 
	    }  
	    if( c == 27 || c == 'q' || c == 'Q' )//按ESC/q/Q退出  
		break;  
	    
	}  

return 0;  
}  


