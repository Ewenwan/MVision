/**
* This file is part of ORB-SLAM2.
*
*/


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>//字符串
#include<thread>// 线程
#include<opencv2/core/core.hpp>// opencv

// user 
#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
    public:
	// Input sensor 枚举  输入传感器类型
	enum eSensor{
	    MONOCULAR=0,// 单目0
	    STEREO=1,    	// 双目1
	    RGBD=2	    	// 深度2
	};

    public:

	// Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
       // 初始化系统  启动 建图 闭环检测  可视化 线程 
	System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true);

	// Proccess the given stereo frame. Images must be synchronized and rectified.
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).  
	// 双目跟踪  返回相机位姿 
	cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

	// Process the given rgbd frame. Depthmap must be registered to the RGB frame.
	// Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Input depthmap: Float (CV_32F).
	// Returns the camera pose (empty if tracking fails).
	// 深度 跟踪  返回相机位姿
	cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);

	// Proccess the given monocular frame
	// Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
	// Returns the camera pose (empty if tracking fails).
	// 单目 跟踪  返回相机位姿
	cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);

	// This stops local mapping thread (map building) and performs only camera tracking.
	// 定位 + 跟踪 模式
	void ActivateLocalizationMode();
	// This resumes local mapping thread and performs SLAM again.
	// 建图 + 跟踪 模式
	void DeactivateLocalizationMode();

	// Returns true if there have been a big map change (loop closure, global BA)
	// since last call to this function
	bool MapChanged();

	// Reset the system (clear map)
	void Reset();

	// All threads will be requested to finish.
	// It waits until all threads have finished.
	// This function must be called before saving the trajectory.
	void Shutdown();

	// Save camera trajectory in the TUM RGB-D dataset format.
	// Only for stereo and RGB-D. This method does not work for monocular.
	// Call first Shutdown()
	// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
	// 保存相机 位姿
	void SaveTrajectoryTUM(const string &filename);

	// Save keyframe poses in the TUM RGB-D dataset format.
	// This method works for all sensor input.
	// Call first Shutdown()
	// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
	void SaveKeyFrameTrajectoryTUM(const string &filename);

	// Save camera trajectory in the KITTI dataset format.
	// Only for stereo and RGB-D. This method does not work for monocular.
	// Call first Shutdown()
	// See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
	void SaveTrajectoryKITTI(const string &filename);

	// TODO: Save/Load functions
	// SaveMap(const string &filename);
	// LoadMap(const string &filename);

	// Information from most recent processed frame
	// You can call this right after TrackMonocular (or stereo or RGBD)
	int GetTrackingState();
	std::vector<MapPoint*> GetTrackedMapPoints();
	std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    private:

	// Input sensor
	eSensor mSensor;// enum 枚举变量  输入相机类型 单目 双目 深度

	// ORB vocabulary used for place recognition and feature matching.
	ORBVocabulary* mpVocabulary;// 词典对象指针 用于 地点识别   特征匹配 orb特征

	// KeyFrame database for place recognition (relocalization and loop detection).
	// 关键帧 数据库 对象指针  用于 地点识别 定位 回环检测
	KeyFrameDatabase* mpKeyFrameDatabase;

	// Map structure that stores the pointers to all KeyFrames and MapPoints.
	// 地图对象指针  存储 关键帧 和 地图点
	Map* mpMap;

	// Tracker. It receives a frame and computes the associated camera pose.
	// It also decides when to insert a new keyframe, create some new MapPoints and
	// performs relocalization if tracking fails.
	// 跟踪对象 指针
	Tracking* mpTracker;

	// Local Mapper. It manages the local map and performs local bundle adjustment.
	// 建图对象 指针
	LocalMapping* mpLocalMapper;

	// Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
	// a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
	// 回环检测对象指针
	LoopClosing* mpLoopCloser;

	// The viewer draws the map and the current camera pose. It uses Pangolin.
	// 可视化对象指针
	Viewer* mpViewer;
	// 画关键帧对象 指针
	FrameDrawer* mpFrameDrawer;
	// 画地图对象 指针
	MapDrawer* mpMapDrawer;

	// System threads: Local Mapping, Loop Closing, Viewer.
	// The Tracking thread "lives" in the main execution thread that creates the System object.
	std::thread* mptLocalMapping;// 建图线程         指针
	std::thread* mptLoopClosing;  // 闭环检测线程  指针
	std::thread* mptViewer;	     // 可视化线程      指针

	// Reset flag 线程重启标志
	std::mutex mMutexReset;
	bool mbReset;

	// Change mode flags
    // 使用std::mutex创建互斥量，通过调用成员函数lock()进行上锁，unlock()进行解锁。但不方便的是需要记住锁后要在函数出口再次调用unlock()解锁. 
    // 因此可以用std::lock_guard,其会在构造的时候提供已锁的互斥量，并在析构的时候进行解锁，从而保证自动管理。
	std::mutex mMutexMode;
	bool mbActivateLocalizationMode;// 跟踪 + 定位
	bool mbDeactivateLocalizationMode;// 跟踪 + 建图

	// Tracking state 跟踪线程 状态
	int mTrackingState;
	std::vector<MapPoint*> mTrackedMapPoints;
	std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
	std::mutex mMutexState;
    };

}// namespace ORB_SLAM

#endif // SYSTEM_H
