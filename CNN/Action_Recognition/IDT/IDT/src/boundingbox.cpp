/*
为视频的每一帧图像手动标注人体区域边框

*/
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace cv;

cv::Mat frame,dst,img,tmp;

float a1,a2,a3,a4,a5=1.0;


void on_mouse(int event,int x,int y, int flags,void * ustc){
    static Point pre_pt = Point(-1,-1);//start-coordinate
    static Point cur_pt = Point(-1,-1);//in-time-coordinate
	char temp[16];
	if (event == CV_EVENT_LBUTTONDOWN)//mouse left down, read start-coordinate,draw circle on it.
	{
	    frame.copyTo(img);//copy original image to img
	    sprintf(temp,"(%d,%d)",x,y);
	    pre_pt = Point(x,y);
	    putText(img,temp,pre_pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0,255),1,8);//show coordinates on window
	    circle(img,pre_pt,2,Scalar(255,0,0,0),CV_FILLED,CV_AA,0);//circling
	    imshow("img",img);
	    }
	else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))//left-mouse dosen't press down
	{
	    img.copyTo(tmp);//copy img to tmp to show in-time coordinate
	    sprintf(temp,"(%d,%d)",x,y);
	    cur_pt = Point(x,y);
	    putText(tmp,temp,cur_pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0,255));//show mouse moving coordinate in-time
	    imshow("img",tmp);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))//left-mouse down, mouse move, draw rectangle
	{
	    img.copyTo(tmp);
	    sprintf(temp,"(%d,%d)",x,y);
	    cur_pt = Point(x,y);
	    putText(tmp,temp,cur_pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0,255));
	    rectangle(tmp,pre_pt,cur_pt,Scalar(0,255,0,0),1,8,0);//show rectangle on tmp image
	    imshow("img",tmp);
	}
	else if (event == CV_EVENT_LBUTTONUP)//left-mouse up, show the rectangle
	{
	    frame.copyTo(img);
	    sprintf(temp,"(%d,%d)",x,y);
	    cur_pt = Point(x,y);
	    putText(img,temp,cur_pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0,255));
	    circle(img,pre_pt,2,Scalar(255,0,0,0),CV_FILLED,CV_AA,0);
	    rectangle(img,pre_pt,cur_pt,Scalar(0,255,0,0),1,8,0);//draw rect on img based on start and end coordinates,
	    imshow("img",img);
	    img.copyTo(tmp);
	    a1=pre_pt.x;
	    a2=pre_pt.y;
	    a3=cur_pt.x;
	    a4=cur_pt.y;
	}
}

int main(int argc, char** argv){
	VideoCapture capture;
	char* video = argv[1];
    capture.open(video);
    if(!capture.isOpened())
	    fprintf(stderr, "Could not initialize capturing..\n");

    int frame_num = 0;
	while(true) {
	    capture >> frame;
        if(frame.empty())
    	    break;
        frame.copyTo(img);
    	frame.copyTo(tmp);
    	namedWindow("img");//define a window
    	setMouseCallback("img",on_mouse,0);//call the callback function
    	imshow("img",img);
        int c=cv::waitKey(0);
        while(c!='q'){
        	printf("%d %f %f %f %f %f\n",frame_num,a1,a2,a3,a4,a5);
        	c=cv::waitKey(0);
        }
    	frame_num++;
	}



	return 0;
	}
