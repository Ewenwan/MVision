/**
* This file is part of ORB-SLAM2.
* ORB主要借鉴了PTAM的思想，借鉴的工作主要有
* Rubble的ORB特征点；
* DBow2的place recognition用于闭环检测；
* Strasdat的闭环矫正和covisibility graph思想；
* 以及Kuemmerle和Grisetti的g2o用于优化。
* 
* 
* 系统入口:
* 1】输入图像    得到 相机位置
*       单目 GrabImageMonocular(im);
*       双目 GrabImageStereo(imRectLeft, imRectRight);
*       深度 GrabImageMonocular(imRectLeft, imRectRight);
* 
* 2】转换为灰度图
*       单目 mImGray
*       双目 mImGray, imGrayRight
*       深度 mImGray, imDepth
* 
* 3】构造 帧Frame
*       单目 未初始化  Frame(mImGray, mpIniORBextractor)
*       单目 已初始化  Frame(mImGray, mpORBextractorLeft)
*       双目      Frame(mImGray, imGrayRight, mpORBextractorLeft, mpORBextractorRight)
*       深度      Frame(mImGray, imDepth,        mpORBextractorLeft)
* 
* 4】跟踪 Track
*   数据流进入 Tracking线程   Tracking.cc
* 
* 
* 
* ORB-SLAM利用三个线程分别进行追踪、地图构建和闭环检测。

一、追踪

    ORB特征提取
    初始姿态估计（速度估计）
    姿态优化（Track local map，利用邻近的地图点寻找更多的特征匹配，优化姿态）
    选取关键帧

二、地图构建

    加入关键帧（更新各种图）
    验证最近加入的地图点（去除Outlier）
    生成新的地图点（三角法）
    局部Bundle adjustment（该关键帧和邻近关键帧，去除Outlier）
    验证关键帧（去除重复帧）

三、闭环检测

    选取相似帧（bag of words）
    检测闭环（计算相似变换（3D<->3D，存在尺度漂移，因此是相似变换），RANSAC计算内点数）
    融合三维点，更新各种图
    图优化（传导变换矩阵），更新地图所有点

* 
* 
*/



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

// wyw添加 2017.11.4
// 字符串搜索
#include <time.h>
bool has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}

namespace ORB_SLAM2
{
	  // 默认初始化函数  单词表文件 txt/bin文件    配置文件     传感器：单目、双目、深度
	  System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,const bool bUseViewer):
			      mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
			      mbDeactivateLocalizationMode(false)//直接初始化变量
	  {
	      // 输出欢迎 信息  Output welcome message
	      cout << endl <<
	      "ORB-SLAM2 单目双目深度相机 SLAM" << endl <<
	      "under certain conditions. See LICENSE.txt." << endl << endl;

	      cout << "输入相机为: ";
	      if(mSensor==MONOCULAR)
		  cout << "单目 Monocular" << endl;
	      else if(mSensor==STEREO)
		  cout << "双目 Stereo" << endl;
	      else if(mSensor==RGBD)
		  cout << "深度 RGB-D" << endl;

	      //Check settings file opencv 读取 配置 文件
	      cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
	      if(!fsSettings.isOpened())
	      {
		cerr << "打不开设置文件 :  " << strSettingsFile << endl;
		exit(-1);
	      }
	      
		  // 配置文件中读取 点云精度 设置 for point cloud resolution
	    //  float resolution = fsSettings["PointCloudMapping.Resolution"];

	      //Load ORB Vocabulary
	      cout << endl << "加载 ORB词典. This could take a while..." << endl;

	      /*
	       使用 new创建对象   类似在 堆空间中申请内存 返回指针
	       使用完后需使用delete删除
	       
	       */
	  //  打开字典文件
	  /////// ////////////////////////////////////
	  //// wyw 修改 2017.11.4
	      clock_t tStart = clock();//时间开始
// 1. 创建字典 mpVocabulary = new ORBVocabulary()；并从文件中载入字典===================================
	      mpVocabulary = new ORBVocabulary();//关键帧字典数据库
	      //bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
	      bool bVocLoad = false; //  bool量  打开字典flag
	      if (has_suffix(strVocFile, ".txt"))//
		    bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);//txt格式打开  
		    // -> 指针对象 的 解引用和 访问成员函数  相当于  (*mpVocabulary).loadFromTextFile(strVocFile);
	      else
		    bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);//bin格式打开
	      if(!bVocLoad)
	      {
		  cerr << "字典路径错误 " << endl;
		  cerr << "打开文件错误: " << strVocFile << endl;
		  exit(-1);
	      }
	      printf("数据库载入时间 Vocabulary loaded in %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);//显示文件载入时间 秒
	  ///////////////////////////////////////////
				      
// 2. 使用特征字典mpVocabulary 创建关键帧数据库 KeyFrameDatabase=========================================
	      mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
				      
// 3. 创建地图对象  mpMap ==============================================================================
	      mpMap = new Map();
				      
// 4. 创建地图显示 帧显示 两个显示窗口 Create Drawers. These are used by the Viewer======================
	      mpFrameDrawer = new FrameDrawer(mpMap);//关键帧显示
	      mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);//地图显示

	      
	          // Initialize pointcloud mapping
	      //mpPointCloudMapping = make_shared<PointCloudMapping>( resolution );

	      //Initialize the Tracking thread
	      //(it will live in the main thread of execution, the one that called this constructor)
// 5. 初始化 跟踪线程 对象 未启动=======================================================================
	      mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
				      mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

	      //Initialize the Local Mapping thread and launch
// 6. 初始化 局部地图构建 线程 并启动线程===============================================================
	      mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
	      mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);

	      //Initialize the Loop Closing thread and launch
// 7. 初始化闭环检测线程 并启动线程====================================================================
	      mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
	      mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

	      //Initialize the Viewer thread and launch
// 8. 初始化 跟踪线程可视化 并启动=====================================================================
	      if(bUseViewer)
	      {
		  mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
		  mptViewer = new thread(&Viewer::Run, mpViewer);
		  mpTracker->SetViewer(mpViewer);
	      }

// 9. 线程之间传递指针 Set pointers between threads===================================================
	      mpTracker->SetLocalMapper(mpLocalMapper);   // 跟踪线程 关联 局部建图和闭环检测线程
	      mpTracker->SetLoopClosing(mpLoopCloser);

	      mpLocalMapper->SetTracker(mpTracker);       // 局部建图线程 关联 跟踪和闭环检测线程
	      mpLocalMapper->SetLoopCloser(mpLoopCloser);

	      mpLoopCloser->SetTracker(mpTracker);        // 闭环检测线程 关联 跟踪和局部建图线程
	      mpLoopCloser->SetLocalMapper(mpLocalMapper);
	  }


	  // 双目跟踪 常 Mat量  左图  右图 时间戳   返回相机位姿 
	  cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
	  {
	      if(mSensor!=STEREO)
	      {
		  cerr << "调用了 双目跟踪 TrackStereo 但是传感器 设置却不是 双目 STEREO." << endl;
		  exit(-1);
	      }   

	      // Check mode change  模式变换的检测  
	      {
		/*
		* 类 unique_lock 是一个一般性质的 mutex 属主的封装，
		* 提供延迟锁定（deferred locking），
		* 限时尝试（time-constrained attempts），
		* 递归锁定（recursive locking），
		* 锁主的转换， 
		* 以及对条件变量的使用。
		*/
		  unique_lock<mutex> lock(mMutexMode);//线程锁定
		  //定位 模式  跟踪+定位   建图关闭
		  if(mbActivateLocalizationMode)
		  {
		      mpLocalMapper->RequestStop();//先停止 建图线程
		      // Wait until Local Mapping has effectively stopped 等待线程 完成 停止
		      while(!mpLocalMapper->isStopped())
		      {
			  usleep(1000);//休息1s
		      }

		      mpTracker->InformOnlyTracking(true);//开启跟踪线程
		      mbActivateLocalizationMode = false;
		  }
		  //非 只定位模式    跟踪 + 定位 +建图
		  if(mbDeactivateLocalizationMode)
		  {
		      mpTracker->InformOnlyTracking(false);//开启跟踪线程
		      mpLocalMapper->Release();// 释放 建图线程   
		      mbDeactivateLocalizationMode = false;
		  }
	      }

	      // Check reset 跟踪线程 重置 检测 
	      {
		  unique_lock<mutex> lock(mMutexReset);
		  if(mbReset)
		  {
		      mpTracker->Reset();// 线程重置
		      mbReset = false;
		  }
	      }

	      cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);//得到 相机位姿

	      unique_lock<mutex> lock2(mMutexState);
	      mTrackingState = mpTracker->mState;//状态
	      mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;//跟踪到的地图点
	      mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;// 
	      return Tcw;
	  }



	  // 深度相机 跟踪
	  cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
	  {
	      if(mSensor!=RGBD)
	      {
		  cerr << "调用了 深度相机跟踪 TrackRGBD 但是传感器却不是深度相机 RGBD." << endl;
		  exit(-1);
	      }    

	      // 切换模式  Check mode change   跟踪+建图    跟踪+定位
	      {
		  unique_lock<mutex> lock(mMutexMode);
		  //  跟踪+定位
		  if(mbActivateLocalizationMode)
		  {
		      mpLocalMapper->RequestStop();//停止 建图
		      // Wait until Local Mapping has effectively stopped
		      while(!mpLocalMapper->isStopped())
		      {
			  usleep(1000);
		      }
		      mpTracker->InformOnlyTracking(true);//仅跟踪
		      mbActivateLocalizationMode = false;
		  }
		// 跟踪+建图
		  if(mbDeactivateLocalizationMode)
		  {
		      mpTracker->InformOnlyTracking(false);// 跟踪 + 建图
		      mpLocalMapper->Release();//释放建图线程
		      mbDeactivateLocalizationMode = false;
		  }
	      }

	      // 检查线程重启 Check reset
	      {
		  unique_lock<mutex> lock(mMutexReset);
		  if(mbReset)
		  {
		      mpTracker->Reset();//线程重置
		      mbReset = false;
		  }
	      }
	      // 初始化-----------------------------
	      // 当前帧 特征点个数 大于500 进行初始化
	      // 设置第一帧为关键帧  位姿为 [I 0] 
	      // 根据第一帧视差求得的深度 计算3D点
	      // 生成地图 添加地图点 地图点观测帧 地图点最好的描述子 更新地图点的方向和距离 
	      //                 关键帧的地图点 当前帧添加地图点  地图添加地图点
	      // 显示地图
	   //  ---- 计算参考帧到当前帧 的变换 Tcr = mTcw  * mTwr---------------
	   
	      // 后面的帧-------------------------------------------------------------------------------------------------
	      // 有运动 则跟踪上一帧 跟踪失败进行 跟踪参考关键帧
	      // 没运动 或者 最近才进行过 重定位 则 跟踪 最近的一个关键帧 参考关键帧
	      // 参考关键帧 跟踪失败 则进行 重定位  跟踪所有关键帧
	      // ----- 跟踪局部地图
	        //  ---- 计算参考帧到当前帧 的变换 Tcr = mTcw  * mTwr---------------
	        
	      cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);//得到相机位姿
	      unique_lock<mutex> lock2(mMutexState);
	      mTrackingState = mpTracker->mState;
	      mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
	      mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
	      return Tcw;
	  }


	  // 单目跟踪
	  cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
	  {
	      if(mSensor!=MONOCULAR)
	      {
		  cerr << "调用了 单目跟踪 TrackMonocular 但是输入的传感器却不是 单目 Monocular." << endl;
		  exit(-1);
	      }

	      // 切换模式  Check mode change   跟踪+建图    跟踪+定位+建图
	      {
		  unique_lock<mutex> lock(mMutexMode);
	//  跟踪+定位 
		  if(mbActivateLocalizationMode)
		  {
		      mpLocalMapper->RequestStop();// 停止 建图
		      // Wait until Local Mapping has effectively stopped
		      while(!mpLocalMapper->isStopped())
		      {
			  usleep(1000);
		      }
		      mpTracker->InformOnlyTracking(true);//仅跟踪
		      mbActivateLocalizationMode = false;
		  }
	 // 跟踪+建图+定位
		  if(mbDeactivateLocalizationMode)
		  {
		      mpTracker->InformOnlyTracking(false);
		      mpLocalMapper->Release();
		      mbDeactivateLocalizationMode = false;
		  }
	      }

	      // 检查线程重启 Check reset
	      {
		  unique_lock<mutex> lock(mMutexReset);
		  if(mbReset)
		  {
		      mpTracker->Reset();
		      mbReset = false;
		  }
	      }
	      // 初始化-------
	      // 连续两帧特征点个数大于100个 且两帧 关键点匹配点对数 大于100个  
	      // 初始帧 [I  0] 第二帧 基础矩阵/单应恢复 [R t] 全局优化  同时得到对应的 3D点
	      // 创建地图 使用 最小化重投影误差BA 进行 地图优化 优化位姿 和地图点
	      // 深度距离中值 倒数 归一化第二帧位姿的 平移向量 和 地图点的 三轴坐标
	      // 显示更新	 
	      // 后面的帧----------
	      
	      cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);// 得到相机位姿
	      unique_lock<mutex> lock2(mMutexState);
	      mTrackingState = mpTracker->mState;
	      mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
	      mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

	      return Tcw;
	  }

	  //  跟踪+定位
	  void System::ActivateLocalizationMode()
	  {
	      unique_lock<mutex> lock(mMutexMode);
	      mbActivateLocalizationMode = true;
	  }

	  // 跟踪+建图
	  void System::DeactivateLocalizationMode()
	  {
	      unique_lock<mutex> lock(mMutexMode);
	      mbDeactivateLocalizationMode = true;
	  }

	  // 地图改变标志
	  bool System::MapChanged()
	  {
	      static int n=0;
	      int curn = mpMap->GetLastBigChangeIdx();
	      if(n<curn)
	      {
		  n=curn;
		  return true;
	      }
	      else
		  return false;
	  }

	  //系统复位重置
	  void System::Reset()
	  {
	      unique_lock<mutex> lock(mMutexReset);
	      mbReset = true;
	  }

	  // 系统关闭  完全关闭所有线程
	  void System::Shutdown()
	  {
	      mpLocalMapper->RequestFinish();// 完成 并关闭建图线程
	      mpLoopCloser->RequestFinish();   // 完成 病关闭闭环检测线程
	      if(mpViewer)//解释 可视化线程
	      {
		  mpViewer->RequestFinish();
		  while(!mpViewer->isFinished())
		      usleep(5000);
	      }
	      // 等待所有线程 完全停止 Wait until all thread have effectively stopped
	      while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
	      {
		  usleep(5000);
	      }
	      if(mpViewer)
		  pangolin::BindToContext("ORB-SLAM2: Map Viewer");
	  }

	  // 保存 TUM数据集 相机位姿 轨迹
	  void System::SaveTrajectoryTUM(const string &filename)
	  {
	      cout << endl << "保存相机位姿轨迹到文件 " << filename << " ..." << endl;
	      // 单目相机
	      if(mSensor==MONOCULAR)
	      {
		  cerr << "单目相机不适合 TUM数据集" << endl;
		  return;
	      }

	      vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();// 关键帧 vector数组容器存储 
	      sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);//排序 保证第一个关键帧在原点
	      cv::Mat Two = vpKFs[0]->GetPoseInverse();//世界坐标系位姿

	      ofstream f;//保存相机位姿轨迹的文件
	      f.open(filename.c_str());
	      f << fixed;
	      // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
	      // which is true when tracking failed (lbL).
	      list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();//关键帧的参考帧(前一帧)
	      list<double>::iterator lT = mpTracker->mlFrameTimes.begin();//时间戳
	      list<bool>::iterator lbL = mpTracker->mlbLost.begin();//标志 跟踪失败
	      for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
		  lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
	      {
		  if(*lbL)//跟踪失败 就跳过
		      continue;

		  KeyFrame* pKF = *lRit;//指针   关键帧的参考帧(前一帧)

		  cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);// 4*4 单位矩阵

		  // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
		  while(pKF->isBad())
		  {
		      Trw = Trw*pKF->mTcp;
		      pKF = pKF->GetParent();//参考帧的 参考帧
		  }

		  Trw = Trw*pKF->GetPose()*Two;//TWO为起始相机位姿

		  cv::Mat Tcw = (*lit)*Trw;// 变换矩阵
		  cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();// 旋转矩阵
		  cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);// 平移向量

		  vector<float> q = Converter::toQuaternion(Rwc);// 旋转矩阵 对应 的四元素
	  // 精度 6 位  时间戳 + 9位精度  平移向量  +9位精度  四元素姿态
		  f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
	      }
	      f.close();
	      cout << endl << "相机位姿 已 保存 trajectory saved!" << endl;
	  }


	  void System::SaveKeyFrameTrajectoryTUM(const string &filename)
	  {
	      cout << endl << "保持关键帧轨迹 Saving keyframe trajectory to " << filename << " ..." << endl;

	      vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();// 关键帧 vector数组容器存储 
	      sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);//排序 保证第一个关键帧在原点

	      // Transform all keyframes so that the first keyframe is at the origin.
	      // After a loop closure the first keyframe might not be at the origin.
	      //cv::Mat Two = vpKFs[0]->GetPoseInverse();

	      ofstream f;
	      f.open(filename.c_str());
	      f << fixed;

	      for(size_t i=0; i<vpKFs.size(); i++)
	      {
		  KeyFrame* pKF = vpKFs[i];//关键帧

		// pKF->SetPose(pKF->GetPose()*Two);

		  if(pKF->isBad())
		      continue;
		// 关键帧的 位姿 已经转化到 第一帧图像坐标系(世界坐标系)
		  cv::Mat R = pKF->GetRotation().t();//旋转矩阵
		  vector<float> q = Converter::toQuaternion(R);//四元素
		  cv::Mat t = pKF->GetCameraCenter();//平移矩阵 当前帧 相机坐标系 中心点位置
		  f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
		    << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

	      }
	      f.close();
	      cout << endl << "trajectory saved!" << endl;
	  }

	  void System::SaveTrajectoryKITTI(const string &filename)
	  {
	      cout << endl << "保持相机位姿  " << filename << " ..." << endl;
	      if(mSensor==MONOCULAR)
	      {
		  cerr << "单目 monocular 不适合 KITTI数据集 " << endl;
		  return;
	      }

	      vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();// 关键帧 vector数组容器存储 
	      sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);//排序 保证第一个关键帧在原点
	      cv::Mat Two = vpKFs[0]->GetPoseInverse();// 世界坐标系位姿

	      ofstream f;
	      f.open(filename.c_str());
	      f << fixed;
	      // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
	      // which is true when tracking failed (lbL).
	      list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();//关键帧的参考值  关键帧位姿已经转化到 世界坐标系下
	      list<double>::iterator lT = mpTracker->mlFrameTimes.begin();//时间戳
	      for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
	      {
		  ORB_SLAM2::KeyFrame* pKF = *lRit;// 参考帧

		  cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);// 变换矩阵 单位矩阵

		  while(pKF->isBad())
		  {
		    //  cout << "bad parent" << endl;
		      Trw = Trw*pKF->mTcp;
		      pKF = pKF->GetParent();
		  }

		  Trw = Trw*pKF->GetPose()*Two;//依次乘以 参考值位姿态

		  cv::Mat Tcw = (*lit)*Trw;
		  cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
		  cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

		  f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
		      Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
		      Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
	      }
	      f.close();
	      cout << endl << "trajectory saved!" << endl;
	  }

	  int System::GetTrackingState()
	  {
	      unique_lock<mutex> lock(mMutexState);
	      return mTrackingState;
	  }

	  vector<MapPoint*> System::GetTrackedMapPoints()
	  {
	      unique_lock<mutex> lock(mMutexState);
	      return mTrackedMapPoints;
	  }

	  vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
	  {
	      unique_lock<mutex> lock(mMutexState);
	      return mTrackedKeyPointsUn;
	  }

} //namespace ORB_SLAM
