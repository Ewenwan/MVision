/**
* This file is part of LSD-SLAM.
* 欧式变换 SE(3) R,t 匹配跟踪
* 这个类是Tracking算法的核心类，里面定义了和刚体运动相关的Traqcking所需要得数据和算法
* 
* 
* 
*/

#pragma once
#include <opencv2/core/core.hpp>
#include "util/settings.h"
#include "util/EigenCoreInclude.h"
#include "util/SophusUtil.h"
#include "Tracking/least_squares.h"


namespace lsd_slam
{

class TrackingReference;// 基于参考帧的 世界点坐标(x,y,z) 点云产生
class Frame;// 帧 像素 梯度 最大梯度值 逆深度  逆深度方差 金字塔

// 欧式变换 SE(3) R,t 匹配跟踪 类
class SE3Tracker
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// 初始层 图像尺寸
	int width, height;
// 相机内参数 矩阵与逆矩阵 参数 与逆参数
	Eigen::Matrix3f K, KInv;
	float fx,fy,cx,cy;
	float fxi,fyi,cxi,cyi;

	DenseDepthTrackerSettings settings;// 系统全局配置文件 稠密深度跟踪器设置类
// 定义了一些Tracking相关的设定，比如最大迭代次数，
// 认为收敛的阈值(百分比形式，有些数据认为98%收敛，有些是99%)，
// 还有huber距离所需参数等

// 调试矩阵 debug images  可视化Tracking的迭代过程 需要的 img
	cv::Mat debugImageResiduals;// 匹配点 灰度匹配误差
	cv::Mat debugImageWeights;
	cv::Mat debugImageSecondFrame;// 跟踪时得第二帧　图像
	cv::Mat debugImageOldImageSource;
	cv::Mat debugImageOldImageWarped;

// 类构造函数
	SE3Tracker(int w, int h, Eigen::Matrix3f K);// 使用图像存储 参数和 相机内参数初始化
	SE3Tracker(const SE3Tracker&) = delete;// 拷贝初始化
// 赋值(=)运算符重载
	SE3Tracker& operator=(const SE3Tracker&) = delete;
// 析构函数
	~SE3Tracker();
// 跟踪帧
	SE3 trackFrame(
			TrackingReference* reference,//跟踪的参考帧
			Frame* frame,// 当前帧
			const SE3& frameToReference_initialEstimate);
	
	SE3 trackFrameOnPermaref(
			Frame* reference,// 帧1
			Frame* frame,// 帧2
			SE3 referenceToFrame);


	float checkPermaRefOverlap(
			Frame* reference,
			SE3 referenceToFrame);

	float pointUsage;
	float lastGoodCount;
	float lastMeanRes;
	float lastBadCount;
	float lastResidual;

	float affineEstimation_a;
	float affineEstimation_b;


	bool diverged;
	bool trackingWasGood;
private:

	float* buf_warped_residual;// 匹配点 像素匹配误差
	float* buf_warped_dx;// 当前帧　亚像素　梯度值　仿射变换后(  乘以　相机内参数)
	float* buf_warped_dy;
	float* buf_warped_x;//  参考帧 3d点 通过R,t变换矩阵 变换到 当前图像坐标系下的坐标值
	float* buf_warped_y;
	float* buf_warped_z;

	float* buf_d;                 // 参考帧 Z轴 倒数  = 参考帧 逆深度
	float* buf_idepthVar;  // 参考帧 逆深度 方差
	float* buf_weight_p;

	int buf_warped_size;// 匹配点数量记录　包括好的匹配点　和　不好的匹配点　数量
	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 根据变换矩阵和原3D点和灰度值　计算匹配点对之间 的　匹配点对像素匹配误差　　以及　匹配点对好坏标志图
	float calcResidualAndBuffers(
			const Eigen::Vector3f* refPoint,
			const Eigen::Vector2f* refColVar,
			int* idxBuf,
			int refNum,
			Frame* frame,
			const Sophus::SE3f& referenceToFrame,
			int level,
			bool plotResidual = false);
// x86架构下 SSE指令集优化
#if defined(ENABLE_SSE)
	float calcResidualAndBuffersSSE(
			const Eigen::Vector3f* refPoint,
			const Eigen::Vector2f* refColVar,
			int* idxBuf,
			int refNum,
			Frame* frame,
			const Sophus::SE3f& referenceToFrame,
			int level,
			bool plotResidual = false);
#endif
// ARM 框架下 NENO指令集优化 
#if defined(ENABLE_NEON)
	float calcResidualAndBuffersNEON(
			const Eigen::Vector3f* refPoint,
			const Eigen::Vector2f* refColVar,
			int* idxBuf,
			int refNum,
			Frame* frame,
			const Sophus::SE3f& referenceToFrame,
			int level,
			bool plotResidual = false);
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
////////////////////////////////////////////////////////////////////////////////////////////
	float calcWeightsAndResidual(
			const Sophus::SE3f& referenceToFrame);
// x86架构下 SSE指令集优化	
#if defined(ENABLE_SSE)
	float calcWeightsAndResidualSSE(
			const Sophus::SE3f& referenceToFrame);
#endif
// ARM 框架下 NENO指令集优化 
#if defined(ENABLE_NEON)
	float calcWeightsAndResidualNEON(
			const Sophus::SE3f& referenceToFrame);
#endif
////////////////////////////////////////////////////////////////////////////////////////////
	
///////////////////////////////////////////////////////////////////////////////////////////
	Vector6 calculateWarpUpdate(
			NormalEquationsLeastSquares &ls);
// x86架构下 SSE指令集优化
#if defined(ENABLE_SSE)
	Vector6 calculateWarpUpdateSSE(
			NormalEquationsLeastSquares &ls);
#endif
// ARM 框架下 NENO指令集优化 
#if defined(ENABLE_NEON)
	Vector6 calculateWarpUpdateNEON(
			NormalEquationsLeastSquares &ls);
#endif
////////////////////////////////////////////////////////////////////////////////////////////////
	
	
// 如果要可视化Tracking的迭代过程，那么第一步自然是把debug相关的参数都设置进去
	void calcResidualAndBuffers_debugStart();
// 完成匹配误差计算  保存误差信息图片	
	void calcResidualAndBuffers_debugFinish(int w);

	// used for image saving
	int iterationNumber;


	float affineEstimation_a_lastIt;
	float affineEstimation_b_lastIt;
};

}
