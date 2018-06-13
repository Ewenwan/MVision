/*
测试视频打开
显示视频并在控制台打印出帧编号的程序。
较老的做法

顺便输出多少帧(这个程序的帧也是从0开始)。

总得来这个程序就是测试一下视频能不能被opencv支持解码出来，
这个代码太简单了.

*/
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <unistd.h>

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

IplImage* image = 0; //当前图像
IplImage* prev_image = 0;//上一帧图像
CvCapture* capture = 0;//视频获取

int show = 1; 

int main( int argc, char** argv )
{
	int frameNum = 0;
// 命令行参数获得视频的地址，
	char* video = argv[1];
	// 再通过地址把视频存到capture结构中。
	capture = cvCreateFileCapture(video);

	if( !capture ) { 
		printf( "Could not initialize capturing..\n" );
		return -1;
	}
	
	if( show == 1 )
		cvNamedWindow( "Video", 0 );
//在用一个while循环读出cpature结构中的frame 帧 图像，
	while( true ) {
		IplImage* frame = 0;//指针
		int i, j, c;
	// get a new frame
		frame = cvQueryFrame( capture );// 读出cpature结构中的frame 帧
		if( !frame )
			break;
	//按照frame的大小和通道造一个img,把frame复制到img中，
		if( !image ) {
			image =  cvCreateImage( cvSize(frame->width,frame->height), 8, 3 );
			image->origin = frame->origin;
		}
		cvCopy( frame, image, 0 );//把frame复制到img中，
	//显示图像窗口
		if( show == 1 ) {
			cvShowImage( "Video", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
		// 打印ID
		std::cerr << "The " << frameNum << "-th frame" << std::endl;
		frameNum++;
	}

	if( show == 1 )
		cvDestroyWindow("Video");

	return 0;
}
