/*
iDT代码的依赖包括两个库：

OpenCV: readme中推荐用2.4.2， 实际上用最新的2.4.13也没问题。
        但OpenCV3就不知道能不能用了，没有试过。
ffmpeg: readme中推荐用0.11.1。实际上装最新的版本也没有问题
这两个库的安装教程网上很多，就不再多做介绍了。
而且也都是很常用的库。

在安装完以上两个库后，就可以进行代码编译了。
只需要在代码文件夹下make一下就好，
编译好的可执行文件在./release/下。

使用时输入 视频文件的路径作为参数即可
./release/DenseTrackStab ./test_sequences/person01_boxing_d1_uncomp.avi。

代码结构
iDT代码中主要包括如下几个代码文件

DenseTrackStab.cpp: iDT算法主程序
DenseTrackStab.h:   轨迹跟踪的一些参数，以及一些数据结构体的定义
Descriptors.h:      特征相关的各种函数
Initialize.h:       初始化相关的各种函数
OpticalFlow.h:      光流相关的各种函数
Video.cpp:          这个程序与iDT算法无关，
                    只是作者提供用来测试两个依赖库是否安装成功的。
					
bound box相关内容
bound box即提供视频帧中人体框的信息，
在计算前后帧的投影变换矩阵时，不使用人体框中的匹配点对。
从而排除人体运动干扰，使得对相机运动的估计更加准确。

作者提供的文件中没有bb_file的格式，
代码中也没有读入bb_file的接口，
若需要用到需要在代码中添加一条读入文件语句
（下面的代码解析中已经添加）。
bb_file的格式如下所示
frame_id a1 a2 a3 a4 a5 b1 b2 b3 b4 b5
其中frame_id是帧的编号，从0开始。
代码中还有检查步骤，保证bb_file的长度与视频的帧数相同。

后面的数据5个一组，为人体框的参数。
按顺序分别为：
框左上角点的x，框左上角点的y，框右下角点的x，框右下角点的y，置信度。
需要注意的是虽然要输入置信度，
但实际上这个置信度在代码里也没有用上的样子，
所以取任意值也不影响使用。

因为一帧图像可能框出来的人有好多个，
这种细粒度的控制比大致框出一个范围能更有效地滤去噪声.

至于如何获得这些bound box的数据，最暴力的方法当然是手工标注，不过这样太辛苦了。
在项目中我们采用了SSD（single shot multibox detector）/yolov3算法检测人体框的位置。算法检测人体框的位置。


主程序代码解析
iDT算法代码的大致思路为：

1. 读入新的一帧
2. 通过SURF特征和光流计算当前帧和上一帧的投影变换矩阵
3. 使用求得的投影变换矩阵对当前帧进行warp变换，消除相机运动影响
4. 利用warp变换后的当前帧图像和上一帧图像计算光流
5. 在各个图像尺度上跟踪轨迹并计算特征
6. 保存当前帧的相关信息，跳到1


几个头文件：
DenseTrackStab.h 定义了Track等的数据结构。最重要的track类里面可以看出：
	std::vector<Point2f> point; //轨迹点
	std::vector<Point2f> disp; //偏移点
	std::vector<float> hog; //hog特征
	std::vector<float> hof; //hof特征
	std::vector<float> mbhX; //mbhX特征
	std::vector<float> mbhY; //mbhY特征
	int index;// 序号
基本方法就是在重采样中提取轨迹，
在轨迹空间中再提取hog,hof,mbh特征，
这些特征组合形成iDT特征，
最终作为这个动作的描述。

Initialize.h：涉及各种数据结构的初始化，usage()可以看看；

OpticalFlow.h: 主要用了Farneback计算光流，
博客参考：
https://blog.csdn.net/ironyoung/article/details/60884929
光流源码：
https://searchcode.com/file/30099587/opencv_source/src/cv/cvoptflowgf.cpp

把金字塔的方法也写进去了，
金字塔方法主要是为了消除不同尺寸的影响，
让描述子有更好的泛化能力。

Descriptors.h：提供一些工具函数:
计算直方图描述子
计算梯度直方图
计算光流直方图
计算光流梯度直方图
密集采样轨迹点  DenseSample
载入 人体边框数据
创建去除人体区域的mask掩膜
对帧图进行单应矩阵反变换 去除相机移动的影响
BFMatcher 计算匹配点对
合并光流匹配点对和 surf匹配点对
根据光流得到光流匹配点

*/
#include "DenseTrackStab.h"//定义了Track等的数据结构。
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>
#include <iostream>
using namespace cv;
using namespace std;

//如果要可视化轨迹，将show_track设置为1
int show_track = 1; 

int main(int argc, char** argv)
{// 读入并打开视频文件
	VideoCapture capture;//opencv
	if (argc<2){ 
		cout << "format ./track video.avi " << endl; 
		return -1;
	}
	char* video = argv[1];
	// 命令行参数解析 Initialize.h
	int flag = arg_parse(argc, argv);
	//fprintf(stdout, "open video   \n");
	cout << "open video: " << argv[1] <<endl;
	capture.open(argv[1]);

	if(!capture.isOpened()) {
		fprintf(stderr, "video open error\n");
		return -1;
	}
	if (argc>2)
    //这句代码是我自己添加的，源代码中没有提供bb_file的输入接口
    char* bb_file = argv[2];
	
	int frame_num = 0;
	
	TrackInfo trackInfo;// 轨迹跟踪信息
	DescInfo hogInfo, hofInfo, mbhInfo;//特征 hog hof MBH
	
 //初始化轨迹信息变量
	InitTrackInfo(&trackInfo, track_length, init_gap);// 轨迹帧序列长度 分割
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);// 8个cell
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);// 9个 cell
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);//8个cell

	SeqInfo seqInfo;// 图像帧序列
	InitSeqInfo(&seqInfo, video);// 把视频信息存到seqInfo当中
	
	//初始化bb信息，将bb_file中的信息加载到bb_list中
	std::vector<Frame> bb_list;//帧 人体mask
	if(bb_file) {
		//
		LoadBoundBox(bb_file, bb_list);
 // 代码中还有检查步骤，保证bb_file的长度与视频的帧数相同。
		assert(bb_list.size() == seqInfo.length);
	}
//这个bb_file的文件格式是
//frameID a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 …
//每行开始由帧序数标出，后面每5个为一组定义一个boudingBox，
// 分别为左上的xy,右下的xy以及置信度，就是对这个矩形的确定概率 
// 因为一帧图像可能框出来的人有好多个，
// 这种细粒度的控制比大致框出一个范围能更有效地滤去噪声
	if(flag)//视频帧数量
		seqInfo.length = end_frame - start_frame + 1;

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)//轨迹显示窗口
		namedWindow("DenseTrackStab", 0);
//初始化surf特征检测器
    //此处200为阈值，数值越小则用于匹配的特征点越多，效果越好（不一定），速度越慢
	SurfFeatureDetector detector_surf(200);//特征点检测
	SurfDescriptorExtractor extractor_surf(true, true);//描述子

	std::vector<Point2f> prev_pts_flow, pts_flow;//光流匹配的点
	std::vector<Point2f> prev_pts_surf, pts_surf;//surf匹配的点
	std::vector<Point2f> prev_pts_all, pts_all;//所有的 综合 光流匹配 和 surf匹配的点

	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;// 关键点
	Mat prev_desc_surf, desc_surf;// 描述子
	Mat flow, human_mask;//光流  人体区域

	Mat image, prev_grey, grey;// 彩色图 灰度/梯度图 
// 金字塔
	std::vector<float> fscales(0);//多尺度
	std::vector<Size> sizes(0);// 
	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	// 记录何时应该计算新的特征点
	int init_counter = 0; // indicate when to detect new feature points
	while(true) 
	{
		Mat frame;
		int i, j, c;

		// get a new frame
		// 读入新的帧
		capture >> frame;
		if(frame.empty())//读取错误 返回退出
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}
		
/*-----------------------对第一帧做处理-------------------------*/
//由于光流需要两帧进行计算，故第一帧不计算光流
		if(frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);// 彩色图
			grey.create(frame.size(), CV_8UC1);//  灰度/梯度图
			prev_grey.create(frame.size(), CV_8UC1);//上一帧灰度/梯度图

			InitPry(frame, fscales, sizes);// 初始化 金字塔

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);// 上一帧灰度图金字塔
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);//总的光流
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);//当前帧反变换后和上一帧做对比提取的光流

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);//
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			xyScaleTracks.resize(scale_num);

			frame.copyTo(image);//彩色图像
			cvtColor(image, prev_grey, CV_BGR2GRAY);//灰度图像
			
    //对于每个图像尺度分别密集采样特征点
			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)// 最大尺度的灰度图
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					// 从上一层金字塔 下采样到下一层图像
				   //尺度0不缩放，其余尺度使用插值方法缩放
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				//  密集采样特征点  dense sampling feature points
				std::vector<Point2f> points(0);// 间隔采样 均匀采样
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				// 保存特征点
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion
			// 背景变换矩阵
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);
            // 人体区域 mask
			human_mask = Mat::ones(frame.size(), CV_8UC1);
			if(bb_file)
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
//human_mask即将人体框外的部分记作1,框内部分记作0
            //在计算surf特征时不计算框内特征（即不使用人身上的特征点做匹配）
			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);//关键点
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);//

			frame_num++;
			continue;
		}


		
/*-----------------------对后续帧做处理-------------------------*/
/*-----------------------从第二帧开始做处理-------------------------*/
//  计算光流 等特征
		init_counter++;
		frame.copyTo(image);// 新一帧图像
		cvtColor(image, grey, CV_BGR2GRAY);// 转换成灰度图

		// 匹配surf特征 来计算 相机移动 match surf features
		// 计算新一帧的surf特征，并与前一帧的surf特帧做匹配
        // surf特征只在图像的原始尺度上计算
		if(bb_file)
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);//得到这一帧的 人体mask
		detector_surf.detect(grey, kpts_surf, human_mask);// 检测去除人体区域的关键点
		extractor_surf.compute(grey, kpts_surf, desc_surf);//计算描述子 
		// 和上一帧做匹配上一帧关键点 当前帧关键点 上一帧关键点描述子 当前帧关键点描述子  得到匹配的关键点对 
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);
	// 得到surf匹配的关键点对  prev_pts_surf, pts_surf
		
		// 在所有尺度上计算光流，并用光流计算前后帧的匹配
		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);// 灰度梯度
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);//计算变换之前的光流
		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);//计算光流匹配
	// 得到光流的匹配 prev_pts_flow, pts_flow
		
	// 结合SURF的匹配 和 光流的匹配
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);
		
	// 用上述点匹配计算前后两帧图像之间的投影变换矩阵H
			//为了避免由于匹配点多数量过少造成 投影变换矩阵计算出错，当匹配很少时直接取单位矩阵作为H
		Mat H = Mat::eye(3, 3, CV_64FC1);//单位矩阵
		if(pts_all.size() > 50) {// 超过50对匹配点
			std::vector<unsigned char> match_mask;//记录内点外点标志
//用来寻找一个最佳的mapping，用这种方法来消除镜头剧烈晃动造成的光流计算不准确的问题。
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)//符合好的映射内点数量超过25 计算的转换矩阵较好
				H = temp;
		}
		// p2 = H*p1
		// p2' = H.inv * p2 剔除相机运动后 场景物体的运动
	// 对当前帧进行warp变换，消除相机运动的影响
		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);// 剔除相机运动后 场景物体的运动
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		// compute optical flow for all scales once
	// 用变换后的图像重新计算各个尺度上的光流图像
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);//计算灰度梯度
		// 计算去除背景运动后的光流 flow_warp_pyr 
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);
		
 // 在每个尺度分别计算特征================================================
		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);//灰度图
			else//尺度0不缩放，其余尺度使用插值方法缩放
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;//当前图像尺寸
			int height = grey_pyr[iScale].rows;

			// 初始化直方图特征 compute the integral histograms
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);//计算HOG 梯度直方图 8
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);//计算光流直方图
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);// 水平光流梯度 直方图
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);// 垂直光流梯度 直方图
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);
			
			
			// 在当前尺度 追踪特征点的轨迹，并计算相关的特征
			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();)
			{
				int index = iTrack->index;//id 轨迹长度 L=15
				
				Point2f prev_point = iTrack->point[index];//上一帧上的点
				// 确保在图像尺寸范围内
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);
				// 加上改点的水平和垂直方向的光流(运动速度)后得到跟踪到的下一帧上对应的点
				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];// 变换之前的光流跟踪点
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);//删除超过范围的点
					continue;
				}
				// 变换之后的光流
				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);
                // 在原始尺度上可视化轨迹
				// draw the trajectories at the first scale
				if(show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

		// 若轨迹的长度达到了预设长度,在iDT中应该是设置为15
				// 达到长度后就可以输出各个特征了
				if(iTrack->index >= trackInfo.length) 
				{
			//1. 轨迹特征 
					std::vector<Point2f> trajectory(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];//归一化尺度后的轨迹 变换之前
				
					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];// 变换之后去除相机移动光流后的光流
	
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);// 轨迹 均值 方差 长度
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement))
					{
						// 打印轨迹 output the trajectory
						printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);

						// for spatio-temporal pyramid
						printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));//归一化到 去除图像大小影响
						printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
						printf("%f\t", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
					
						// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i)
							printf("%f\t%f\t", displacement[i].x, displacement[i].y);
						// 打印 直方图特征
						//实际上，traj特征的效果一般，可以去掉，那么输出以下几个就好了
						//如果需要保存输出的特征，可以修改PrintDesc函数
						PrintDesc(iTrack->hog, hogInfo, trackInfo);
						PrintDesc(iTrack->hof, hofInfo, trackInfo);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
						printf("\n");
					}

					iTrack = tracks.erase(iTrack);//释放 这个跟踪轨迹资源
					continue;
				}
				++iTrack;
			}
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);
			//下采样
			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);// 迭代
		//每帧处理完后保存该帧信息，用作下一帧计算时用
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);//灰度图金字塔 5层
			poly_pyr[i].copyTo(prev_poly_pyr[i]);//光流图金字塔
		}

		prev_kpts_surf = kpts_surf;// 上一帧 SURF关键点
		desc_surf.copyTo(prev_desc_surf);// 对应的描述子

		frame_num++;//帧数++
		
// 可视化窗口
		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
	}
// 
	if( show_track == 1 )
		destroyWindow("DenseTrackStab");//释放窗口资源

	return 0;
}
