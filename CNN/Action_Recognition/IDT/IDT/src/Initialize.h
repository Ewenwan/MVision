/*
Initialize.h：涉及各种数据结构的初始化，usage()可以看看；
//命令行参数解析
bool arg_parse(int argc, char** argv){}

*/
#ifndef INITIALIZE_H_
#define INITIALIZE_H_

#include "DenseTrackStab.h"

using namespace cv;
//轨迹信息
void InitTrackInfo(TrackInfo* trackInfo, int track_length, int init_gap)
{
	trackInfo->length = track_length;
	trackInfo->gap = init_gap;
}
//申请直方图 内存空间
DescMat* InitDescMat(int height, int width, int nBins)
{
	DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));//申请内存空间
	descMat->height = height;
	descMat->width = width;
	descMat->nBins = nBins;//0~360分割的bin数量

	long size = height*width*nBins;//总长度
	//数据指针 申请内存空间
	descMat->desc = (float*)malloc(size*sizeof(float));
	// 清零
	memset(descMat->desc, 0, size*sizeof(float));
	return descMat;
}
//清理直方图 内存空间
void ReleDescMat(DescMat* descMat)
{
	free(descMat->desc);
	free(descMat);
}
//初始化直方图描述
void InitDescInfo(DescInfo* descInfo, int nBins, bool isHof, int size, int nxy_cell, int nt_cell)
{
	descInfo->nBins = nBins;//直方图分割bin  8/9
	descInfo->isHof = isHof;
	descInfo->nxCells = nxy_cell;//2*2cell 32*32像素
	descInfo->nyCells = nxy_cell;
	descInfo->ntCells = nt_cell;//15帧分3份 /5帧
	descInfo->dim = nBins*nxy_cell*nxy_cell;
	descInfo->height = size;
	descInfo->width = size;
}
//  视频信息 打开视频 获取帧宽高 以及帧数量
void InitSeqInfo(SeqInfo* seqInfo, char* video)
{
	VideoCapture capture;
	capture.open(video);//打开视频

	if(!capture.isOpened())
		fprintf(stderr, "Could not initialize capturing..\n");

	// get the number of frames in the video
	int frame_num = 0;
	while(true) {
		Mat frame;
		capture >> frame;//获取每一帧图片

		if(frame.empty())
			break;

		if(frame_num == 0) {//初始化帧宽高
			seqInfo->width = frame.cols;
			seqInfo->height = frame.rows;
		}

		frame_num++;//记录帧数量
    }
	seqInfo->length = frame_num;
}

// 使用信息
void usage()
{
	fprintf(stderr, "Extract improved trajectories from a video\n\n");//提取密集轨迹特征
	fprintf(stderr, "Usage: DenseTrackStab video_file [options]\n");// 命令行参数
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -h                        Display this message and exit\n");// 打印本帮助信息，并退出
	fprintf(stderr, "  -S [start frame]          The start frame to compute feature (default: S=0 frame)\n");//开始帧id
	fprintf(stderr, "  -E [end frame]            The end frame for feature computing (default: E=last frame)\n");//结束帧id
	fprintf(stderr, "  -L [trajectory length]    The length of the trajectory (default: L=15 frames)\n");//轨迹长度
	fprintf(stderr, "  -W [sampling stride]      The stride for dense sampling feature points (default: W=5 pixels)\n");//密集间隔采样间隔
	fprintf(stderr, "  -N [neighborhood size]    The neighborhood size for computing the descriptor (default: N=32 pixels)\n");//像素框 尺寸
	fprintf(stderr, "  -s [spatial cells]        The number of cells in the nxy axis (default: nxy=2 cells)\n");// 像素框分割子框 2*2
	fprintf(stderr, "  -t [temporal cells]       The number of cells in the nt axis (default: nt=3 cells)\n");// 轨迹时间长度分段
	fprintf(stderr, "  -A [scale number]         The number of maximal spatial scales (default: 8 scales)\n");// 直方图分割bin
	fprintf(stderr, "  -I [initial gap]          The gap for re-sampling feature points (default: 1 frame)\n");//采样帧间隔
	fprintf(stderr, "  -H [human bounding box]   The human bounding box file to remove outlier matches (default: None)\n");// 人体区域边框文件
}
//命令行参数解析
bool arg_parse(int argc, char** argv)
{
	int c;
	bool flag = false;
	char* executable = basename(argv[0]);//可执行文件名
	while((c = getopt (argc, argv, "hS:E:L:W:N:s:t:A:I:H:")) != -1)
	switch(c) {
		case 'S':
		start_frame = atoi(optarg);//字符串转成整形 //开始帧id
		flag = true;
		break;
		case 'E':
		end_frame = atoi(optarg);//结束帧id
		flag = true;
		break;
		case 'L':
		track_length = atoi(optarg);//轨迹长度
		break;
		case 'W':
		min_distance = atoi(optarg);//密集间隔采样间隔
		break;
		case 'N':
		patch_size = atoi(optarg);//像素框 尺寸
		break;
		case 's':
		nxy_cell = atoi(optarg);// 像素框分割子框 2*2
		break;
		case 't':
		nt_cell = atoi(optarg);// 轨迹时间长度分段
		break;
		case 'A':
		scale_num = atoi(optarg);// 直方图分割bin
		break;
		case 'I':
		init_gap = atoi(optarg);//采样帧间隔
		break;	
		case 'H':
		bb_file = optarg;// 人体区域边框文件 字符串
		break;
		case 'h':
		usage();//打印帮助信息
		exit(0);//退出
		break;

		default:
		fprintf(stderr, "error parsing arguments at -%c\n  Try '%s -h' for help.", c, executable );
		usage();//打印帮助信息
		abort();
	}
	return flag;//重新定义 起始帧和结束帧
}

#endif /*INITIALIZE_H_*/
