/**
* This file is part of LSD-SLAM.
* 帧  位置+姿态 类结构体
*/

#pragma once
#include "util/SophusUtil.h"
#include "GlobalMapping/g2oTypeSim3Sophus.h"



namespace lsd_slam
{
class Frame;// 使用Frame类
class FramePoseStruct {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW// Eigen 矩阵对其 
        // 构造函数
	FramePoseStruct(Frame* frame);
	// 析构函数
	virtual ~FramePoseStruct();

	// parent, the frame originally tracked on. never changes.
	// 父亲帧
	FramePoseStruct* trackingParent;

	// set initially as tracking result (then it's a SE(3)),
	// and is changed only once, when the frame becomes a KF (->rescale).
	// sR,t
	Sim3 thisToParent_raw;


	int frameID;//帧ID
	Frame* frame;// 帧


	// whether this poseStruct is registered in the Graph. if true MEMORY WILL BE HANDLED BY GRAPH
	bool isRegisteredToGraph;//  注册到 地图

	// whether pose is optimized (true only for KF, after first applyPoseGraphOptResult())
	bool isOptimized;// 

	// true as soon as the vertex is added to the g2o graph.
	bool isInGraph;

	// graphVertex (if the frame has one, i.e. is a KF and has been added to the graph, otherwise 0).
	VertexSim3* graphVertex;

	void setPoseGraphOptResult(Sim3 camToWorld);
	void applyPoseGraphOptResult();
	Sim3 getCamToWorld(int recursionDepth = 0);
	void invalidateCache();
private:
	int cacheValidFor;
	static int cacheValidCounter;

	// absolute position (camToWorld).
	// can change when optimization offset is merged.
	Sim3 camToWorld;

	// new, optimized absolute position. is added on mergeOptimization.
	Sim3 camToWorld_new;

	// whether camToWorld_new is newer than camToWorld
	bool hasUnmergedPose;
};

}
