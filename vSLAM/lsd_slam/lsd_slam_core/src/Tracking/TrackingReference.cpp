/**
* This file is part of LSD-SLAM.
* 跟踪的第一步就是 为当前帧 旋转 跟踪的参考帧
* 
* 
* 
* 
* 
* 
*/

#include "Tracking/TrackingReference.h"
#include "DataStructures/Frame.h"// 帧类 图像金字塔 梯度金字塔 逆深度金字塔等
#include "DepthEstimation/DepthMapPixelHypothesis.h"// 逆深度初始化  高斯分布均值初始化 等
#include "GlobalMapping/KeyFrameGraph.h"// 全局地图
#include "util/globalFuncs.h"
#include "IOWrapper/ImageDisplay.h"// 图像显示

namespace lsd_slam
{
// 类初始化函数
TrackingReference::TrackingReference()
{
	frameID=-1;
	keyframe = 0;
	wh_allocated = 0;
	for (int level = 0; level < PYRAMID_LEVELS; ++ level)
	{
		posData[level] = nullptr;
		gradData[level] = nullptr;
		colorAndVarData[level] = nullptr;
		pointPosInXYGrid[level] = nullptr;
		numData[level] = 0;
	}
}

// 清除帧上所有的 每一金字塔层级对应的 坐标点 梯度 像素值 方差等信息 对应的内存
void TrackingReference::releaseAll()
{
	for (int level = 0; level < PYRAMID_LEVELS; ++ level)// 遍历每一个 金字塔层级
	{
		if(posData[level] != nullptr) delete[] posData[level];// delete[]  删除指针数组 位置数据
		if(gradData[level] != nullptr) delete[] gradData[level];// 梯度数据
		if(colorAndVarData[level] != nullptr) delete[] colorAndVarData[level];// 颜色和方差数据
		if(pointPosInXYGrid[level] != nullptr)//位置对应的网格数据
			Eigen::internal::aligned_free((void*)pointPosInXYGrid[level]);
		numData[level] = 0;// 清理 计数
	}
	wh_allocated = 0;// 第 0 层金字塔 像素总数 这里全部清理了 所以为0 
}
// 清理 点云 计数
void TrackingReference::clearAll()
{
	for (int level = 0; level < PYRAMID_LEVELS; ++ level)
		numData[level] = 0;
}

//  析构函数
TrackingReference::~TrackingReference()
{
	boost::unique_lock<boost::mutex> lock(accessMutex);
	invalidate();//关键帧内存锁 解锁
	releaseAll();// 清除帧上所有的 每一金字塔层级对应的 坐标点 梯度 像素值 方差等信息 对应的内存
}

// 导入 帧 更新帧的资源配置的记录
void TrackingReference::importFrame(Frame* sourceKF)
{
  // 上锁   在这个时候 只有本程序能够使用 相应的内存资源
	boost::unique_lock<boost::mutex> lock(accessMutex);
  // 获取锁
	keyframeLock = sourceKF->getActiveLock();
	keyframe = sourceKF;// 帧
	frameID=keyframe->id();// 身份证 编号
// 判断是否需要重置
// 如果宽度和高度的乘积和先前的高度与宽度的乘积一样大，那么就不用重新配置了
	// reset allocation if dimensions differ (shouldnt happen usually)
	if(sourceKF->width(0) * sourceKF->height(0) != wh_allocated)
	{
		releaseAll();// 如果不同，那么就释放先前分配的资源
	       // 然后把资源配置的记录 记录成当前的配置数
		wh_allocated = sourceKF->width(0) * sourceKF->height(0);
	}
	clearAll();// 最后再清空 点云的记录数据numData
// 内存解锁
	lock.unlock();
}


// 关键帧内存锁 解锁
void TrackingReference::invalidate()
{
	if(keyframe != 0)
		keyframeLock.unlock();// 解锁
	keyframe = 0;
}
// 
void TrackingReference::makePointCloud(int level)
{
	assert(keyframe != 0);
	boost::unique_lock<boost::mutex> lock(accessMutex);

	if(numData[level] > 0)
		return;	// already exists.

	int w = keyframe->width(level);
	int h = keyframe->height(level);

	float fxInvLevel = keyframe->fxInv(level);
	float fyInvLevel = keyframe->fyInv(level);
	float cxInvLevel = keyframe->cxInv(level);
	float cyInvLevel = keyframe->cyInv(level);

	const float* pyrIdepthSource = keyframe->idepth(level);
	const float* pyrIdepthVarSource = keyframe->idepthVar(level);
	const float* pyrColorSource = keyframe->image(level);
	const Eigen::Vector4f* pyrGradSource = keyframe->gradients(level);

	if(posData[level] == nullptr) posData[level] = new Eigen::Vector3f[w*h];
	if(pointPosInXYGrid[level] == nullptr)
		pointPosInXYGrid[level] = (int*)Eigen::internal::aligned_malloc(w*h*sizeof(int));;
	if(gradData[level] == nullptr) gradData[level] = new Eigen::Vector2f[w*h];
	if(colorAndVarData[level] == nullptr) colorAndVarData[level] = new Eigen::Vector2f[w*h];

	Eigen::Vector3f* posDataPT = posData[level];
	int* idxPT = pointPosInXYGrid[level];
	Eigen::Vector2f* gradDataPT = gradData[level];
	Eigen::Vector2f* colorAndVarDataPT = colorAndVarData[level];

	for(int x=1; x<w-1; x++)
		for(int y=1; y<h-1; y++)
		{
			int idx = x + y*w;

			if(pyrIdepthVarSource[idx] <= 0 || pyrIdepthSource[idx] == 0) continue;

			*posDataPT = (1.0f / pyrIdepthSource[idx]) * Eigen::Vector3f(fxInvLevel*x+cxInvLevel,fyInvLevel*y+cyInvLevel,1);
			*gradDataPT = pyrGradSource[idx].head<2>();
			*colorAndVarDataPT = Eigen::Vector2f(pyrColorSource[idx], pyrIdepthVarSource[idx]);
			*idxPT = idx;

			posDataPT++;
			gradDataPT++;
			colorAndVarDataPT++;
			idxPT++;
		}

	numData[level] = posDataPT - posData[level];
}

}
