/**
* This file is part of LSD-SLAM.
* 跟踪的第一步就是 为当前帧 旋转 跟踪的参考帧
*  包含 每一层金字塔上的：
* 1.   关键点位置坐标                                posData[i]                 (x,y,z)
* 2.   关键点像素梯度信息                        gradData[i]                (dx, dy)
* 3.   关键点像素值 和 逆深度方差信息   colorAndVarData[i]   (I, Var)
* 4.   关键点位置对应的 灰度像素点        pointPosInXYGrid[i]  x + y*width
*       上面四个都是指针数组
* 5.   产生的 物理世界中的点的数量       numData[i]
* 
*/

#pragma once
#include "util/settings.h"
#include "util/EigenCoreInclude.h"
#include "boost/thread/mutex.hpp"
#include <boost/thread/shared_mutex.hpp>


namespace lsd_slam
{

class Frame;// 帧类 图像金字塔 梯度金字塔 逆深度金字塔等                        在DataStructures下
class DepthMapPixelHypothesis;// 逆深度初始化  高斯分布均值初始化 等 在 DepthEstimation 下
class KeyFrameGraph;// 关键帧地图                                                                在GlobalMapping 下

/**
 * Point cloud used to track frame poses.
 */

// 跟踪参考帧 类
class TrackingReference
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW// 使用EIGEN 一般都需要加的 宏

	/** Creates an empty TrackingReference with optional preallocation per level. */
	// 类初始化函数
	TrackingReference();
	// 类析构函数
	~TrackingReference();
	//  导入 帧 更新帧的资源配置的记录
	void importFrame(Frame* source);
        // 关键帧
	Frame* keyframe;
       //  关键帧 内存锁
	boost::shared_lock<boost::shared_mutex> keyframeLock;
	// 帧ID 身份证 编号
	int frameID;
	
        // 创建点云
	void makePointCloud(int level);
	
	// 清理 计数 numData[level] = 0;  内存 未清理
	void clearAll();
	void invalidate();
	// 每一个金字塔层级上2d像素点 对应的 3d 位置
	Eigen::Vector3f* posData[PYRAMID_LEVELS];	// (x,y,z)
	// 每一个金字塔层级上2d像素点 对应的 两个方向的梯度信息
	Eigen::Vector2f* gradData[PYRAMID_LEVELS];	// (dx, dy)
	// 每一个金字塔层级上2d像素点 对应的 像素值和 方差信息
	Eigen::Vector2f* colorAndVarData[PYRAMID_LEVELS];	// (I, Var)
	// 关键点位置对应的 灰度像素点        
	int* pointPosInXYGrid[PYRAMID_LEVELS];	// x + y*width
	int numData[PYRAMID_LEVELS];// 点云的记录数据

private:
	int wh_allocated;//  第 0 层金字塔 像素总数    资源配置的记录
	boost::mutex accessMutex;
	void releaseAll();// 清除帧上所有的 每一金字塔层级对应的 坐标点 梯度 像素值 方差等信息 对应的内存
};
}
