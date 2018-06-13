#ifndef DENSETRACKSTAB_H_
#define DENSETRACKSTAB_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
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

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

int start_frame = 0;
int end_frame = INT_MAX;
int scale_num = 8;//金字塔层级数量
const float scale_stride = sqrt(2);
char* bb_file = NULL;

// parameters for descriptors
int patch_size = 32;// 光流点附件的 区域尺寸
int nxy_cell = 2;//32*32的区域 分成2*2个小格子
int nt_cell = 3;
float epsilon = 0.05;
const float min_flow = 0.4;

// parameters for tracking
double quality = 0.001;
int min_distance = 5;
int init_gap = 1;//间隔
int track_length = 15;//轨迹长度 15帧

// parameters for rejecting trajectory
const float min_var = sqrt(3);
const float max_var = 50;
const float max_dis = 20;

//矩形框结构体 左上角点+宽高
typedef struct {
	int x;       // top left corner
	int y;
	int width;
	int height;
}RectInfo;
// 视频信息
typedef struct {
    int width;   // resolution of the video
    int height;
    int length;  // number of frames
}SeqInfo;
//轨迹信息
typedef struct {
    int length;  // length of the trajectory
    int gap;     // initialization gap for feature re-sampling 
}TrackInfo;
//直方图信息
typedef struct {
    int nBins;   // number of bins for vector quantization
    bool isHof; 
    int nxCells; // number of cells in x direction
    int nyCells; 
    int ntCells;
    int dim;     // dimension of the descriptor
    int height;  // size of the block for computing the descriptor
    int width;
}DescInfo; 
//直方图
// integral histogram for the descriptors
typedef struct {
    int height;
    int width;
    int nBins;//0~360分割的bin数量
    float* desc;//数据指针
}DescMat;

//轨迹
class Track
{
public:
    std::vector<Point2f> point;//轨迹点
    std::vector<Point2f> disp;//偏移点
    std::vector<float> hog;//hog特征 梯度直方图
    std::vector<float> hof;//hof特征 光流直方图
    std::vector<float> mbhX;//mbhX特征光流梯度直方图
    std::vector<float> mbhY; //mbhY特征
    int index;

    Track(const Point2f& point_, const TrackInfo& trackInfo, const DescInfo& hogInfo,
          const DescInfo& hofInfo, const DescInfo& mbhInfo)
        : point(trackInfo.length+1), 
		  disp(trackInfo.length), 
		  hog(hogInfo.dim*trackInfo.length),
          hof(hofInfo.dim*trackInfo.length), 
		  mbhX(mbhInfo.dim*trackInfo.length), 
		  mbhY(mbhInfo.dim*trackInfo.length)
    {
        index = 0;
        point[0] = point_;
    }

    void addPoint(const Point2f& point_)
    {
        index++;
        point[index] = point_;
    }
};

//人体区域边框
class BoundBox
{
public:
	Point2f TopLeft;//上左
	Point2f BottomRight;//下右
	float confidence;//置信度

	BoundBox(float a1, float a2, float a3, float a4, float a5)
	{
		TopLeft.x = a1;
		TopLeft.y = a2;
		BottomRight.x = a3;
		BottomRight.y = a4;
		confidence = a5;//置信度
	}
};
//帧 类
class Frame
{
public:
	int frameID;//id
	std::vector<BoundBox> BBs;//包含的人体区域边框 数组
	//构造函数
	Frame(const int& frame_)
	{
		frameID = frame_;
		BBs.clear();//清空
	}
};

#endif /*DENSETRACKSTAB_H_*/
