      /**
      lsd是一个大规模的单目视觉半稠密slam项目。

      博客参考解析  https://blog.csdn.net/lancelot_vim

      http://www.cnblogs.com/hitcm/category/763753.html

      https://blog.csdn.net/xdEddy/article/details/78009748

      https://blog.csdn.net/u013004597

      https://blog.csdn.net/tiandijun/article/details/62226163

      官网:http://vision.in.tum.de/research/vslam/lsdslam
      代码:https://github.com/tum-vision/lsd_slam

      运行lsd-slam
      一个来自官方的范例，使用的dataset如下，400+M
      http://vmcremers8.informatik.tu-muenchen.de/lsd/LSD_room.bag.zip
      解压之
      然后运行下面的3个命令，即可看到效果
      rosrun lsd_slam_viewer viewer
      rosrun lsd_slam_core live_slam image:=/image_raw camera_info:=/camera_info
      rosbag play ./LSD_room.bag

      平移，旋转，相似以及投影变换，在lsd-slam中，有个三方开源库叫做Sophus/sophus，封装好了前三个变换。
      库分析  Sophus/sophushttps://blog.csdn.net/lancelot_vim/article/details/51706832


      算法整体框架
      1. Tracking 跟踪线程，当前图像与当前关键帧匹配，获取姿态变换；
      2. 深度图估计线程，    创建新的关键帧/优化当前关键帧，更新关键帧数据库
					    创建新的关键帧： 传播深度信息到新的关键帧，正则化深度图
					    优化当前关键帧：近似为小基线长度双目，概率卡尔曼滤波优化更新，正则化深度图
      3. 全局地图优化，        关键帧加入当地图，从地图中匹配最相似的关键帧，估计sim3位姿变换


      */

      #include "SlamSystem.h"

      #include "DataStructures/Frame.h"//帧 数据结构
      #include "Tracking/SE3Tracker.h"// 欧式变换跟踪 R t
      #include "Tracking/Sim3Tracker.h"//相似变换sR t 跟踪 关心的是尺度统一性
      #include "DepthEstimation/DepthMap.h"//深度估计 获取R t之后基线搜索 三角测量获取深度
      #include "Tracking/TrackingReference.h"//跟踪参考关键帧
      #include "LiveSLAMWrapper.h"
      #include "util/globalFuncs.h"
      #include "GlobalMapping/KeyFrameGraph.h"// 全局建图 关键帧地图
      #include "GlobalMapping/TrackableKeyFrameSearch.h"//搜索可跟踪的关键帧
      #include "GlobalMapping/g2oTypeSim3Sophus.h"// smi3 优化
      #include "IOWrapper/ImageDisplay.h"// 显示图像
      #include "IOWrapper/Output3DWrapper.h"//输出3d点云地图
      #include <g2o/core/robust_kernel_impl.h>// G2o图优化
      #include "DataStructures/FrameMemory.h"//数据结构   内存管理             1
      #include "deque"// 双端队列

      // for mkdir
      #include <sys/types.h>
      #include <sys/stat.h>

      #ifdef ANDROID
      #include <android/log.h>
      #endif

      #include "opencv2/opencv.hpp"

      using namespace lsd_slam;

      // 类初始化函数
      SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, bool enableSLAM)
      : SLAMEnabled(enableSLAM), relocalizer(w,h,K)
      {
	      if(w%16 != 0 || h%16!=0)
	      {
		      printf("image dimensions must be multiples of 16! Please crop your images / video accordingly.\n");
		      assert(false);
	      }

	      this->width = w;// 帧宽度
	      this->height = h;//帧高度
	      this->K = K;//　相机内参数
	      trackingIsGood = true;


	      currentKeyFrame =  nullptr;
	      trackingReferenceFrameSharedPT = nullptr;
	      keyFrameGraph = new KeyFrameGraph();
	      createNewKeyFrame = false;

	      map =  new DepthMap(w,h,K);// 逆深度图
	      
	      newConstraintAdded = false;
	      haveUnmergedOptimizationOffset = false;


	      tracker = new SE3Tracker(w,h,K);//欧式变换矩阵　跟踪对象
	      // Do not use more than 4 levels for odometry tracking
	      for (int level = 4; level < PYRAMID_LEVELS; ++level)
		      tracker->settings.maxItsPerLvl[level] = 0;// 层级越高　图像越小　跟踪精度越低
	      trackingReference = new TrackingReference();//跟踪　参考帧对象
	      mappingTrackingReference = new TrackingReference();//建图参考帧　对象

// 使能定位和建图
	      if(SLAMEnabled)
	      {
		      trackableKeyFrameSearch = new TrackableKeyFrameSearch(keyFrameGraph,w,h,K);//可跟踪的　关键帧
		      constraintTracker = new Sim3Tracker(w,h,K);// 相似变换跟踪　建图　回环检测时　的　跟踪
		      constraintSE3Tracker = new SE3Tracker(w,h,K);// 欧式变换跟踪
		      newKFTrackingReference = new TrackingReference();// 新的关键帧
		      candidateTrackingReference = new TrackingReference();// 候选参考帧
	      }
	      else
	      {
		      constraintSE3Tracker = 0;
		      trackableKeyFrameSearch = 0;
		      constraintTracker = 0;
		      newKFTrackingReference = 0;
		      candidateTrackingReference = 0;
	      }


	      outputWrapper = 0;

	      keepRunning = true;
	      doFinalOptimization = false;
	      depthMapScreenshotFlag = false;
	      lastTrackingClosenessScore = 0;

	      thread_mapping = boost::thread(&SlamSystem::mappingThreadLoop, this);

	      if(SLAMEnabled)
	      {
		      thread_constraint_search = boost::thread(&SlamSystem::constraintSearchThreadLoop, this);
		      thread_optimization = boost::thread(&SlamSystem::optimizationThreadLoop, this);
	      }



	      msTrackFrame = msOptimizationIteration = msFindConstraintsItaration = msFindReferences = 0;
	      nTrackFrame = nOptimizationIteration = nFindConstraintsItaration = nFindReferences = 0;
	      nAvgTrackFrame = nAvgOptimizationIteration = nAvgFindConstraintsItaration = nAvgFindReferences = 0;
	      gettimeofday(&lastHzUpdate, NULL);

      }
// 类析构函数
      SlamSystem::~SlamSystem()
      {
	      keepRunning = false;

	      // make sure none is waiting for something.
	      printf("... waiting for SlamSystem's threads to exit\n");
	      newFrameMappedSignal.notify_all();
	      unmappedTrackedFramesSignal.notify_all();
	      newKeyFrameCreatedSignal.notify_all();
	      newConstraintCreatedSignal.notify_all();

	      thread_mapping.join();
	      thread_constraint_search.join();
	      thread_optimization.join();
	      printf("DONE waiting for SlamSystem's threads to exit\n");

	      if(trackableKeyFrameSearch != 0) delete trackableKeyFrameSearch;
	      if(constraintTracker != 0) delete constraintTracker;
	      if(constraintSE3Tracker != 0) delete constraintSE3Tracker;
	      if(newKFTrackingReference != 0) delete newKFTrackingReference;
	      if(candidateTrackingReference != 0) delete candidateTrackingReference;

	      delete mappingTrackingReference;
	      delete map;
	      delete trackingReference;
	      delete tracker;

	      // make shure to reset all shared pointers to all frames before deleting the keyframegraph!
	      unmappedTrackedFrames.clear();
	      latestFrameTriedForReloc.reset();
	      latestTrackedFrame.reset();
	      currentKeyFrame.reset();
	      trackingReferenceFrameSharedPT.reset();

	      // delte keyframe graph
	      delete keyFrameGraph;

	      FrameMemory::getInstance().releaseBuffes();

	      Util::closeAllWindows();
      }
// 可视化
      void SlamSystem::setVisualization(Output3DWrapper* outputWrapper)
      {
	      this->outputWrapper = outputWrapper;
      }

      void SlamSystem::mergeOptimizationOffset()
      {
	      // update all vertices that are in the graph!
	      poseConsistencyMutex.lock();

	      bool needPublish = false;
	      if(haveUnmergedOptimizationOffset)
	      {
		      keyFrameGraph->keyframesAllMutex.lock_shared();
		      for(unsigned int i=0;i<keyFrameGraph->keyframesAll.size(); i++)
			      keyFrameGraph->keyframesAll[i]->pose->applyPoseGraphOptResult();
		      keyFrameGraph->keyframesAllMutex.unlock_shared();

		      haveUnmergedOptimizationOffset = false;
		      needPublish = true;
	      }

	      poseConsistencyMutex.unlock();


	      if(needPublish)
		      publishKeyframeGraph();
      }

/*
 LSD-SLAM构建的是半稠密逆深度地图（semi-dense inverse depth map），
 只对有明显梯度的像素位置进行深度估计，用逆深度表示，并且假设逆深度服从高斯分布。
 一旦一个图像帧被选为关键帧，则用其跟踪的参考帧的深度图对其进行深度图构建，
 之后跟踪到该新建关键帧的图像帧都会用来对其深度图进行更新。
 当然，追溯到第一帧，肯定是没有深度图的，
 因此第一帧的深度图是有明显梯度区域随机生成的深度。
 
总的来说，建图线程可以分为两种情况

   1.  构建关键帧，则使用参考帧的深度图对新帧构建新的深度图（Depth Map Creation）(深度图的传播)
   2. 不构建关键帧，则更新参考关键帧的深度图（Depth Map Refinement）()
   
正常情况下，每次跟踪一次图像之后，建图线程都会调用一次SlamSystem::doMappingIteration() 
SlamSystem::doMappingIteration()  函数就是整个建图线程的主体函数。
 */
// 建图循环线程
      void SlamSystem::mappingThreadLoop()
      {
	      printf("Started mapping thread!\n");
	      while(keepRunning)
	      {
		      if (!doMappingIteration())
		      {
			      boost::unique_lock<boost::mutex> lock(unmappedTrackedFramesMutex);
			      unmappedTrackedFramesSignal.timed_wait(lock,boost::posix_time::milliseconds(200));	// slight chance of deadlock otherwise
			      lock.unlock();
		      }

		      newFrameMappedMutex.lock();
		      newFrameMappedSignal.notify_all();
		      newFrameMappedMutex.unlock();
	      }
	      printf("Exited mapping thread \n");
      }

      void SlamSystem::finalize()
      {
	      printf("Finalizing Graph... finding final constraints!!\n");

	      lastNumConstraintsAddedOnFullRetrack = 1;
	      while(lastNumConstraintsAddedOnFullRetrack != 0)
	      {
		      doFullReConstraintTrack = true;
		      usleep(200000);
	      }


	      printf("Finalizing Graph... optimizing!!\n");
	      doFinalOptimization = true;
	      newConstraintMutex.lock();
	      newConstraintAdded = true;
	      newConstraintCreatedSignal.notify_all();
	      newConstraintMutex.unlock();
	      while(doFinalOptimization)
	      {
		      usleep(200000);
	      }


	      printf("Finalizing Graph... publishing!!\n");
	      unmappedTrackedFramesMutex.lock();
	      unmappedTrackedFramesSignal.notify_one();
	      unmappedTrackedFramesMutex.unlock();
	      while(doFinalOptimization)
	      {
		      usleep(200000);
	      }
	      boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
	      newFrameMappedSignal.wait(lock);
	      newFrameMappedSignal.wait(lock);

	      usleep(200000);
	      printf("Done Finalizing Graph.!!\n");
      }

// 建图一致性约束也就是做闭环检测和全局优化。
// 其默认检测闭环的方式是帧与帧之间做双向跟踪。
      void SlamSystem::constraintSearchThreadLoop()
      {
	      printf("Started  constraint search thread!\n");
	      
	      boost::unique_lock<boost::mutex> lock(newKeyFrameMutex);
	      int failedToRetrack = 0;

	      while(keepRunning)
	      {
//1.  新关键帧队列为空，则在所有关键帧中随机选取测试闭环
      //关于如何选择候选帧，主要步骤如下：
      //视角和距离判别
      //SE3跟踪检测
      //Sim3跟踪检测
		      if(newKeyFrames.size() == 0)
		      {
			      lock.unlock();
			      keyFrameGraph->keyframesForRetrackMutex.lock();
			      bool doneSomething = false;
			      if(keyFrameGraph->keyframesForRetrack.size() > 10)
			      {
				      std::deque< Frame* >::iterator toReTrack = keyFrameGraph->keyframesForRetrack.begin() + (rand() % (keyFrameGraph->keyframesForRetrack.size()/3));
				      Frame* toReTrackFrame = *toReTrack;

				      keyFrameGraph->keyframesForRetrack.erase(toReTrack);
				      keyFrameGraph->keyframesForRetrack.push_back(toReTrackFrame);

				      keyFrameGraph->keyframesForRetrackMutex.unlock();
                                     // 测试闭环 findConstraintsForNewKeyFrames
				      int found = findConstraintsForNewKeyFrames(toReTrackFrame, false, false, 2.0);
				      if(found == 0)
					      failedToRetrack++;
				      else
					      failedToRetrack=0;

				      if(failedToRetrack < (int)keyFrameGraph->keyframesForRetrack.size() - 5)
					      doneSomething = true;
			      }
			      else
				      keyFrameGraph->keyframesForRetrackMutex.unlock();

			      lock.lock();

			      if(!doneSomething)
			      {
				      if(enablePrintDebugInfo && printConstraintSearchInfo)
					      printf("nothing to re-track... waiting.\n");
				      newKeyFrameCreatedSignal.timed_wait(lock,boost::posix_time::milliseconds(500));

			      }
		      }
// 2. 新关键帧队列不为空，则取最早的新关键帧测试闭环
		      else
		      {
			      Frame* newKF = newKeyFrames.front();
			      newKeyFrames.pop_front();
			      lock.unlock();

			      struct timeval tv_start, tv_end;
			      gettimeofday(&tv_start, NULL);

			      findConstraintsForNewKeyFrames(newKF, true, true, 1.0);
			      failedToRetrack=0;
			      gettimeofday(&tv_end, NULL);
			      msFindConstraintsItaration = 0.9*msFindConstraintsItaration + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
			      nFindConstraintsItaration++;

			      FrameMemory::getInstance().pruneActiveFrames();
			      lock.lock();
		      }


		      if(doFullReConstraintTrack)
		      {
			      lock.unlock();
			      printf("Optizing Full Map!\n");

			      int added = 0;
			      for(unsigned int i=0;i<keyFrameGraph->keyframesAll.size();i++)
			      {
				      if(keyFrameGraph->keyframesAll[i]->pose->isInGraph)
					      added += findConstraintsForNewKeyFrames(keyFrameGraph->keyframesAll[i], false, false, 1.0);
			      }

			      printf("Done optizing Full Map! Added %d constraints.\n", added);

			      doFullReConstraintTrack = false;

			      lastNumConstraintsAddedOnFullRetrack = added;
			      lock.lock();
		      }
	      }

	      printf("Exited constraint search thread \n");
      }
// 而图优化的部分在另外一个线程SlamSystem::optimizationThreadLoop，基本就是使用g2o
      void SlamSystem::optimizationThreadLoop()
      {
	      printf("Started optimization thread \n");

	      while(keepRunning)
	      {
		      boost::unique_lock<boost::mutex> lock(newConstraintMutex);
		      if(!newConstraintAdded)
			      newConstraintCreatedSignal.timed_wait(lock,boost::posix_time::milliseconds(2000));	// slight chance of deadlock otherwise
		      newConstraintAdded = false;
		      lock.unlock();

		      if(doFinalOptimization)
		      {
			      printf("doing final optimization iteration!\n");
			      optimizationIteration(50, 0.001);
			      doFinalOptimization = false;
		      }
		      while(optimizationIteration(5, 0.02));
	      }

	      printf("Exited optimization thread \n");
      }

      void SlamSystem::publishKeyframeGraph()
      {
	      if (outputWrapper != nullptr)
		      outputWrapper->publishKeyframeGraph(keyFrameGraph);
      }

      void SlamSystem::requestDepthMapScreenshot(const std::string& filename)
      {
	      depthMapScreenshotFilename = filename;
	      depthMapScreenshotFlag = true;
      }
// 每当构造完一个关键帧都会调用，做了填补当前关键帧深度以及平滑深度图的工作，
// 把关键帧设置为函数中就给图像帧设置了在关键帧中的编号idxInKeyframes。
// 以及把当前关键帧currentKeyFrame加入关键帧集合keyFrameGraph->keyframesAll中等工作。
      void SlamSystem::finishCurrentKeyframe()
      {
	      if(enablePrintDebugInfo && printThreadingInfo)
		      printf("FINALIZING KF %d\n", currentKeyFrame->id());

	      map->finalizeKeyFrame();

	      if(SLAMEnabled)
	      {
		      mappingTrackingReference->importFrame(currentKeyFrame.get());
		      currentKeyFrame->setPermaRef(mappingTrackingReference);
		      mappingTrackingReference->invalidate();

		      if(currentKeyFrame->idxInKeyframes < 0)
		      {
			      keyFrameGraph->keyframesAllMutex.lock();
			      currentKeyFrame->idxInKeyframes = keyFrameGraph->keyframesAll.size();
			      keyFrameGraph->keyframesAll.push_back(currentKeyFrame.get());
			      keyFrameGraph->totalPoints += currentKeyFrame->numPoints;
			      keyFrameGraph->totalVertices ++;
			      keyFrameGraph->keyframesAllMutex.unlock();

			      newKeyFrameMutex.lock();
			      newKeyFrames.push_back(currentKeyFrame.get());
			      newKeyFrameCreatedSignal.notify_all();
			      newKeyFrameMutex.unlock();
		      }
	      }

	      if(outputWrapper!= 0)
		      outputWrapper->publishKeyframe(currentKeyFrame.get());
      }

      void SlamSystem::discardCurrentKeyframe()
      {
	      if(enablePrintDebugInfo && printThreadingInfo)
		      printf("DISCARDING KF %d\n", currentKeyFrame->id());

	      if(currentKeyFrame->idxInKeyframes >= 0)
	      {
		      printf("WARNING: trying to discard a KF that has already been added to the graph... finalizing instead.\n");
		      finishCurrentKeyframe();
		      return;
	      }


	      map->invalidate();

	      keyFrameGraph->allFramePosesMutex.lock();
	      for(FramePoseStruct* p : keyFrameGraph->allFramePoses)
	      {
		      if(p->trackingParent != 0 && p->trackingParent->frameID == currentKeyFrame->id())
			      p->trackingParent = 0;
	      }
	      keyFrameGraph->allFramePosesMutex.unlock();


	      keyFrameGraph->idToKeyFrameMutex.lock();
	      keyFrameGraph->idToKeyFrame.erase(currentKeyFrame->id());
	      keyFrameGraph->idToKeyFrameMutex.unlock();

      }

      void SlamSystem::createNewCurrentKeyframe(std::shared_ptr<Frame> newKeyframeCandidate)
      {
	      if(enablePrintDebugInfo && printThreadingInfo)
		      printf("CREATE NEW KF %d from %d\n", newKeyframeCandidate->id(), currentKeyFrame->id());


	      if(SLAMEnabled)
	      {
		      // add NEW keyframe to id-lookup
		      keyFrameGraph->idToKeyFrameMutex.lock();
		      keyFrameGraph->idToKeyFrame.insert(std::make_pair(newKeyframeCandidate->id(), newKeyframeCandidate));
		      keyFrameGraph->idToKeyFrameMutex.unlock();
	      }

	      // propagate & make new.
	      map->createKeyFrame(newKeyframeCandidate.get());

	      if(printPropagationStatistics)
	      {

		      Eigen::Matrix<float, 20, 1> data;
		      data.setZero();
		      data[0] = runningStats.num_prop_attempts / ((float)width*height);
		      data[1] = (runningStats.num_prop_created + runningStats.num_prop_merged) / (float)runningStats.num_prop_attempts;
		      data[2] = runningStats.num_prop_removed_colorDiff / (float)runningStats.num_prop_attempts;

		      outputWrapper->publishDebugInfo(data);
	      }

	      currentKeyFrameMutex.lock();
	      currentKeyFrame = newKeyframeCandidate;
	      currentKeyFrameMutex.unlock();
      }
      void SlamSystem::loadNewCurrentKeyframe(Frame* keyframeToLoad)
      {
	      if(enablePrintDebugInfo && printThreadingInfo)
		      printf("RE-ACTIVATE KF %d\n", keyframeToLoad->id());

	      map->setFromExistingKF(keyframeToLoad);

	      if(enablePrintDebugInfo && printRegularizeStatistics)
		      printf("re-activate frame %d!\n", keyframeToLoad->id());

	      currentKeyFrameMutex.lock();
	      currentKeyFrame = keyFrameGraph->idToKeyFrame.find(keyframeToLoad->id())->second;
	      currentKeyFrame->depthHasBeenUpdatedFlag = false;
	      currentKeyFrameMutex.unlock();
      }
// 个函数用来改变当前关键帧currentKeyFrame。
// 如果在地图中存在与当前候选关键帧很相似的关键帧，则使用地图中已有的关键帧，否则重新构建关键帧。
      void SlamSystem::changeKeyframe(bool noCreate, bool force, float maxScore)
      {
	      Frame* newReferenceKF=0;
	      std::shared_ptr<Frame> newKeyframeCandidate = latestTrackedFrame;
	      if(doKFReActivation && SLAMEnabled)
	      {
		      struct timeval tv_start, tv_end;
		      gettimeofday(&tv_start, NULL);
		      newReferenceKF = trackableKeyFrameSearch->findRePositionCandidate(newKeyframeCandidate.get(), maxScore);
		      gettimeofday(&tv_end, NULL);
		      msFindReferences = 0.9*msFindReferences + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		      nFindReferences++;
	      }

	      if(newReferenceKF != 0)
		      loadNewCurrentKeyframe(newReferenceKF);
	      else
	      {
		      if(force)
		      {
			      if(noCreate)
			      {
				      trackingIsGood = false;
				      nextRelocIdx = -1;
				      printf("mapping is disabled & moved outside of known map. Starting Relocalizer!\n");
			      }
			      else
				      createNewCurrentKeyframe(newKeyframeCandidate);
		      }
	      }
	      createNewKeyFrame = false;
      }
// 更新参考关键帧深度
      bool SlamSystem::updateKeyframe()
      {
	      std::shared_ptr<Frame> reference = nullptr;
	      std::deque< std::shared_ptr<Frame> > references;
// 变量unmappedTrackedFrames是SlamSystem::trackFrame中使用了，是用来保存图像帧的队列:
	      unmappedTrackedFramesMutex.lock();// 是记录了所有的图像帧。

	      // remove frames that have a different tracking parent.
	      while(unmappedTrackedFrames.size() > 0 &&
			      (!unmappedTrackedFrames.front()->hasTrackingParent() ||
					      unmappedTrackedFrames.front()->getTrackingParent() != currentKeyFrame.get()))
	      {
		      unmappedTrackedFrames.front()->clear_refPixelWasGood();
		      //把所有不是跟踪到当前关键帧的图像帧都从队列中剔除了
		      unmappedTrackedFrames.pop_front();
	      }

	      // clone list
	      if(unmappedTrackedFrames.size() > 0)
	      {
		      for(unsigned int i=0;i<unmappedTrackedFrames.size(); i++)
			// 接下来把跟踪到当前关键帧的图像帧都放在 references 中
			      references.push_back(unmappedTrackedFrames[i]);

		      std::shared_ptr<Frame> popped = unmappedTrackedFrames.front();
		      // 并且把最老的图像帧从队列中删除
		      unmappedTrackedFrames.pop_front();
		      unmappedTrackedFramesMutex.unlock();

		      if(enablePrintDebugInfo && printThreadingInfo)
			      printf("MAPPING %d on %d to %d (%d frames)\n", currentKeyFrame->id(), references.front()->id(), references.back()->id(), (int)references.size());
// 接下来就是更新地图的函数DepthMap::updateKeyframe，把所有跟踪到当前关键帧的图像帧用于建图。
		      map->updateKeyframe(references);

		      popped->clear_refPixelWasGood();
		      references.clear();
	      }
	      else
	      {
		      unmappedTrackedFramesMutex.unlock();
		      return false;
	      }


	      if(enablePrintDebugInfo && printRegularizeStatistics)
	      {
		      Eigen::Matrix<float, 20, 1> data;
		      data.setZero();
		      data[0] = runningStats.num_reg_created;
		      data[2] = runningStats.num_reg_smeared;
		      data[3] = runningStats.num_reg_deleted_secondary;
		      data[4] = runningStats.num_reg_deleted_occluded;
		      data[5] = runningStats.num_reg_blacklisted;

		      data[6] = runningStats.num_observe_created;
		      data[7] = runningStats.num_observe_create_attempted;
		      data[8] = runningStats.num_observe_updated;
		      data[9] = runningStats.num_observe_update_attempted;


		      data[10] = runningStats.num_observe_good;
		      data[11] = runningStats.num_observe_inconsistent;
		      data[12] = runningStats.num_observe_notfound;
		      data[13] = runningStats.num_observe_skip_oob;
		      data[14] = runningStats.num_observe_skip_fail;

		      outputWrapper->publishDebugInfo(data);
	      }



	      if(outputWrapper != 0 && continuousPCOutput && currentKeyFrame != 0)
		      outputWrapper->publishKeyframe(currentKeyFrame.get());

	      return true;
      }


      void SlamSystem::addTimingSamples()
      {
	      map->addTimingSample();
	      struct timeval now;
	      gettimeofday(&now, NULL);
	      float sPassed = ((now.tv_sec-lastHzUpdate.tv_sec) + (now.tv_usec-lastHzUpdate.tv_usec)/1000000.0f);
	      if(sPassed > 1.0f)
	      {
		      nAvgTrackFrame = 0.8*nAvgTrackFrame + 0.2*(nTrackFrame / sPassed); nTrackFrame = 0;
		      nAvgOptimizationIteration = 0.8*nAvgOptimizationIteration + 0.2*(nOptimizationIteration / sPassed); nOptimizationIteration = 0;
		      nAvgFindReferences = 0.8*nAvgFindReferences + 0.2*(nFindReferences / sPassed); nFindReferences = 0;

		      if(trackableKeyFrameSearch != 0)
		      {
			      trackableKeyFrameSearch->nAvgTrackPermaRef = 0.8*trackableKeyFrameSearch->nAvgTrackPermaRef + 0.2*(trackableKeyFrameSearch->nTrackPermaRef / sPassed); trackableKeyFrameSearch->nTrackPermaRef = 0;
		      }
		      nAvgFindConstraintsItaration = 0.8*nAvgFindConstraintsItaration + 0.2*(nFindConstraintsItaration / sPassed); nFindConstraintsItaration = 0;
		      nAvgOptimizationIteration = 0.8*nAvgOptimizationIteration + 0.2*(nOptimizationIteration / sPassed); nOptimizationIteration = 0;

		      lastHzUpdate = now;


		      if(enablePrintDebugInfo && printOverallTiming)
		      {
			      printf("MapIt: %3.1fms (%.1fHz); Track: %3.1fms (%.1fHz); Create: %3.1fms (%.1fHz); FindRef: %3.1fms (%.1fHz); PermaTrk: %3.1fms (%.1fHz); Opt: %3.1fms (%.1fHz); FindConst: %3.1fms (%.1fHz);\n",
					      map->msUpdate, map->nAvgUpdate,
					      msTrackFrame, nAvgTrackFrame,
					      map->msCreate+map->msFinalize, map->nAvgCreate,
					      msFindReferences, nAvgFindReferences,
					      trackableKeyFrameSearch != 0 ? trackableKeyFrameSearch->msTrackPermaRef : 0, trackableKeyFrameSearch != 0 ? trackableKeyFrameSearch->nAvgTrackPermaRef : 0,
					      msOptimizationIteration, nAvgOptimizationIteration,
					      msFindConstraintsItaration, nAvgFindConstraintsItaration);
		      }
	      }

      }


      void SlamSystem::debugDisplayDepthMap()
      {


	      map->debugPlotDepthMap();
	      double scale = 1;
	      if(currentKeyFrame != 0 && currentKeyFrame != 0)
		      scale = currentKeyFrame->getScaledCamToWorld().scale();
	      // debug plot depthmap
	      char buf1[200];
	      char buf2[200];


	      snprintf(buf1,200,"Map: Upd %3.0fms (%2.0fHz); Trk %3.0fms (%2.0fHz); %d / %d / %d",
			      map->msUpdate, map->nAvgUpdate,
			      msTrackFrame, nAvgTrackFrame,
			      currentKeyFrame->numFramesTrackedOnThis, currentKeyFrame->numMappedOnThis, (int)unmappedTrackedFrames.size());

	      snprintf(buf2,200,"dens %2.0f%%; good %2.0f%%; scale %2.2f; res %2.1f/; usg %2.0f%%; Map: %d F, %d KF, %d E, %.1fm Pts",
			      100*currentKeyFrame->numPoints/(float)(width*height),
			      100*tracking_lastGoodPerBad,
			      scale,
			      tracking_lastResidual,
			      100*tracking_lastUsage,
			      (int)keyFrameGraph->allFramePoses.size(),
			      keyFrameGraph->totalVertices,
			      (int)keyFrameGraph->edgesAll.size(),
			      1e-6 * (float)keyFrameGraph->totalPoints);


	      if(onSceenInfoDisplay)
		      printMessageOnCVImage(map->debugImageDepth, buf1, buf2);
	      if (displayDepthMap)
		      Util::displayImage( "DebugWindow DEPTH", map->debugImageDepth, false );

	      int pressedKey = Util::waitKey(1);
	      handleKey(pressedKey);
      }


      void SlamSystem::takeRelocalizeResult()
      {
	      Frame* keyframe;
	      int succFrameID;
	      SE3 succFrameToKF_init;
	      std::shared_ptr<Frame> succFrame;
	      relocalizer.stop();
	      relocalizer.getResult(keyframe, succFrame, succFrameID, succFrameToKF_init);
	      assert(keyframe != 0);

	      loadNewCurrentKeyframe(keyframe);

	      currentKeyFrameMutex.lock();
	      trackingReference->importFrame(currentKeyFrame.get());
	      trackingReferenceFrameSharedPT = currentKeyFrame;
	      currentKeyFrameMutex.unlock();

	      tracker->trackFrame(
			      trackingReference,
			      succFrame.get(),
			      succFrameToKF_init);

	      if(!tracker->trackingWasGood || tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount) < 1-0.75f*(1-MIN_GOODPERGOODBAD_PIXEL))
	      {
		      if(enablePrintDebugInfo && printRelocalizationInfo)
			      printf("RELOCALIZATION FAILED BADLY! discarding result.\n");
		      trackingReference->invalidate();
	      }
	      else
	      {
		      keyFrameGraph->addFrame(succFrame.get());

		      unmappedTrackedFramesMutex.lock();
		      if(unmappedTrackedFrames.size() < 50)
			// 是用来保存图像帧的队列:
			      unmappedTrackedFrames.push_back(succFrame);
		      unmappedTrackedFramesMutex.unlock();

		      currentKeyFrameMutex.lock();
		      createNewKeyFrame = false;
		      trackingIsGood = true;
		      currentKeyFrameMutex.unlock();
	      }
      }
// 由此看，正常情况下，每次跟踪一次图像之后，建图线程都会调用一次SlamSystem::doMappingIteration()。
// 该函数就是整个建图线程的主体函数。
// 当前关键帧变量currentKeyFrame都是通过函数SlamSystem::changeKeyframe来改变的，
// 但还没有加入关键帧集合集合keyFrameGraph->keyframesAll，
// 新构建的关键帧只有通过函数SlamSystem::finishCurrentKeyframe之后才算是真正的关键帧。
      bool SlamSystem::doMappingIteration()
      {
	      if(currentKeyFrame == 0)
		      return false;
	      if(!doMapping && currentKeyFrame->idxInKeyframes < 0)
	      {
		//每当构造完一个关键帧都会调用，做了填补当前关键帧深度以及平滑深度图的工作，
		// 把关键帧设置为函数中就给图像帧设置了在关键帧中的编号idxInKeyframes。
		//以及把当前关键帧currentKeyFrame加入关键帧集合keyFrameGraph->keyframesAll中等工作。
		      if(currentKeyFrame->numMappedOnThisTotal >= MIN_NUM_MAPPED)
			      finishCurrentKeyframe();
		      else
	// 由于在函数SlamSystem::trackFrame中把每一帧图像都加入了图keyFrameGraph（这个有点像关键帧候选队列），
	// 因此该函数主要作用就是把该关键帧直接从keyFrameGraph中剔除。
			      discardCurrentKeyframe();

		      map->invalidate();
		      printf("Finished KF %d as Mapping got disabled!\n",currentKeyFrame->id());
           // 当前关键帧变量currentKeyFrame都是通过函数SlamSystem::changeKeyframe来改变的
		      changeKeyframe(true, true, 1.0f);
	      }

	      mergeOptimizationOffset();
	      addTimingSamples();

	      if(dumpMap)
	      {
		      keyFrameGraph->dumpMap(packagePath+"/save");
		      dumpMap = false;
	      }


	      // set mappingFrame
	      if(trackingIsGood)
	      {
// 如果跟踪线程跟踪都很好，则进入建图部分，否则进行重定位
		      if(!doMapping)
		      {
			      //printf("tryToChange refframe, lastScore %f!\n", lastTrackingClosenessScore);
			      if(lastTrackingClosenessScore > 1)
				      changeKeyframe(true, false, lastTrackingClosenessScore * 0.75);

			      if (displayDepthMap || depthMapScreenshotFlag)
				      debugDisplayDepthMap();

			      return false;
		      }

		      if (createNewKeyFrame)
		      {
             // 在跟踪线程触发了构建新的关键帧，则把当前的关键帧currentKeyFrame处理好，然后把当前帧构建为currentKeyFrame
			 // 深度图传播
			      finishCurrentKeyframe();// 把当前的关键帧currentKeyFrame处理好
			      changeKeyframe(false, true, 1.0f);// 然后把当前帧构建为currentKeyFrame
			      if (displayDepthMap || depthMapScreenshotFlag)
				      debugDisplayDepthMap();
		      }
             // 否则则更新当前关键帧的深度图　　深度图更新
		      else
		      {
			      bool didSomething = updateKeyframe();

			      if (displayDepthMap || depthMapScreenshotFlag)
				      debugDisplayDepthMap();
			      if(!didSomething)
				      return false;
		      }

		      return true;
	      }
	      else
	      {
// 否则进行重定位
		      // invalidate map if it was valid.
		      if(map->isValid())
		      {
			      if(currentKeyFrame->numMappedOnThisTotal >= MIN_NUM_MAPPED)
				      finishCurrentKeyframe();
			      else
				      discardCurrentKeyframe();

			      map->invalidate();
		      }

		      // start relocalizer if it isnt running already
		      if(!relocalizer.isRunning)
			      relocalizer.start(keyFrameGraph->keyframesAll);

		      // did we find a frame to relocalize with?
		      if(relocalizer.waitResult(50))
			      takeRelocalizeResult();


		      return true;
	      }
      }


      void SlamSystem::gtDepthInit(uchar* image, float* depth, double timeStamp, int id)
      {
	      printf("Doing GT initialization!\n");

	      currentKeyFrameMutex.lock();

	      currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, image));
	      currentKeyFrame->setDepthFromGroundTruth(depth);

	      map->initializeFromGTDepth(currentKeyFrame.get());
	      keyFrameGraph->addFrame(currentKeyFrame.get());

	      currentKeyFrameMutex.unlock();

	      if(doSlam)
	      {
		      keyFrameGraph->idToKeyFrameMutex.lock();
		      keyFrameGraph->idToKeyFrame.insert(std::make_pair(currentKeyFrame->id(), currentKeyFrame));
		      keyFrameGraph->idToKeyFrameMutex.unlock();
	      }
	      if(continuousPCOutput && outputWrapper != 0) outputWrapper->publishKeyframe(currentKeyFrame.get());

	      printf("Done GT initialization!\n");
      }


      void SlamSystem::randomInit(uchar* image, double timeStamp, int id)
      {
	      printf("Doing Random initialization!\n");

	      if(!doMapping)
		      printf("WARNING: mapping is disabled, but we just initialized... THIS WILL NOT WORK! Set doMapping to true.\n");


	      currentKeyFrameMutex.lock();

	      currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, image));
	      map->initializeRandomly(currentKeyFrame.get());
	      keyFrameGraph->addFrame(currentKeyFrame.get());

	      currentKeyFrameMutex.unlock();

	      if(doSlam)
	      {
		      keyFrameGraph->idToKeyFrameMutex.lock();
		      keyFrameGraph->idToKeyFrame.insert(std::make_pair(currentKeyFrame->id(), currentKeyFrame));
		      keyFrameGraph->idToKeyFrameMutex.unlock();
	      }
	      if(continuousPCOutput && outputWrapper != 0) outputWrapper->publishKeyframe(currentKeyFrame.get());


	      if (displayDepthMap || depthMapScreenshotFlag)
		      debugDisplayDepthMap();


	      printf("Done Random initialization!\n");

      }
/*
 LSD-SLAM的Tracking 算法
 
 这个函数的代码主要分为如下几个步骤实现：

    1. 构造新的图像帧：把当前图像构建为新的图像帧
    2. 更新参考帧：        如果当前参考帧不是最近的关键帧，则更新参考帧
    　　　　　　　　　 参考帧指的是当前帧的参考帧就一个
    　　　　　　　　　 关键帧很多,每隔一段距离就会产生
    　　　　　　　　　 将与当前帧最近的关键帧选择为参考帧
    3. 初始化位姿：        把上一帧与参考帧的位姿当做初始位姿 R,t
    4. SE3求解：             调用SE3Tracker计算当前帧和参考帧间的位姿变换
    5. 判断是否跟踪失败：根据跟踪的像素点个数多少以及跟踪质量来判断
    6. 关键帧筛选：           通过计算得分确定是否构造新的关键帧
输入：
　　　image　　当前帧图像指针
　　　frameID　 帧id
　　    blockUntilMapped  标志
　　    timestamp　　　　时间戳
 */
      void SlamSystem::trackFrame(uchar* image, unsigned int frameID, bool blockUntilMapped, double timestamp)
      {
	      // Create new frame
// 第一步.  创建新的一帧　会创建　金字塔图　[像素 梯度 最大梯度值 逆深度  逆深度方差]
	      std::shared_ptr<Frame> trackingNewFrame(new Frame(frameID, width, height, K, timestamp, image));

	      if(!trackingIsGood)
	      {
		// 更新当前帧
		      relocalizer.updateCurrentFrame(trackingNewFrame);

		      unmappedTrackedFramesMutex.lock();
		      unmappedTrackedFramesSignal.notify_one();
		      unmappedTrackedFramesMutex.unlock();
		      return;
	      }

	      currentKeyFrameMutex.lock();
	      bool my_createNewKeyframe = createNewKeyFrame;	// pre-save here, to make decision afterwards.
// 第二步. 如果当前参考帧不是最近的关键帧，则更新参考帧
	      if(trackingReference->keyframe != currentKeyFrame.get() || currentKeyFrame->depthHasBeenUpdatedFlag)
	      {
		      trackingReference->importFrame(currentKeyFrame.get());// 更新最近的关键帧 为 参考帧
		      currentKeyFrame->depthHasBeenUpdatedFlag = false;// 当前最近的关键帧深度未更新
		      trackingReferenceFrameSharedPT = currentKeyFrame;//当前参考帧也即最近的关键帧
	      }

	      FramePoseStruct* trackingReferencePose = trackingReference->keyframe->pose;
	      currentKeyFrameMutex.unlock();
	      // DO TRACKING & Show tracking result.
	      if(enablePrintDebugInfo && printThreadingInfo)
		      printf("TRACKING %d on %d\n", trackingNewFrame->id(), trackingReferencePose->frameID);// 当前帧 参考帧
// 第三步.  初始化位姿：把上一帧与参考帧的位姿当做初始位姿
	      poseConsistencyMutex.lock_shared();
	      SE3 frameToReference_initialEstimate = se3FromSim3(
		// 上一帧到世界 * 世界到上一帧的参考帧＝上一帧到参考帧的位姿
			      trackingReferencePose->getCamToWorld().inverse() * keyFrameGraph->allFramePoses.back()->getCamToWorld());
	      poseConsistencyMutex.unlock_shared();
	      struct timeval tv_start, tv_end;
	      gettimeofday(&tv_start, NULL);
// 第四步.  SE3求解：调用SE3Tracker计算当前帧和参考帧间的位姿变换  加权LM算法优化　最小化参考帧3d点反投影到当前帧的光度误差函数
	      SE3 newRefToFrame_poseUpdate = tracker->trackFrame(
			      trackingReference,//参考帧
			      trackingNewFrame.get(),// 当前帧
			      frameToReference_initialEstimate);//上一帧的位姿，初始化为当前帧的初始位姿

	      gettimeofday(&tv_end, NULL);
	      msTrackFrame = 0.9*msTrackFrame + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	      nTrackFrame++;

	      tracking_lastResidual = tracker->lastResidual;//光度误差
	      tracking_lastUsage = tracker->pointUsage;// 深度改变均值
	      // 好的匹配点占总的匹配点比例　　　总的像素点　＝　好的匹配点 + 不好的匹配点＋为能匹配的点 
	      // 可以理解为当前帧和参考帧重叠区域的比例。
	      tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
	      // 好的匹配点占总的像素点比例
	      tracking_lastGoodPerTotal = tracker->lastGoodCount / (trackingNewFrame->width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));

// 第五步.  判断是否跟踪失败：根据跟踪的像素点个数多少以及跟踪质量来判断
	      if(manualTrackingLossIndicated || tracker->diverged || (keyFrameGraph->keyframesAll.size() > INITIALIZATION_PHASE_COUNT && !tracker->trackingWasGood))
	      {
		// 打印跟踪匹配信息
		      printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
				      trackingNewFrame->id(),
				      100*tracking_lastGoodPerTotal,
				      100*tracking_lastGoodPerBad,
				      tracker->diverged ? "DIVERGED" : "NOT DIVERGED");

		      trackingReference->invalidate();

		      trackingIsGood = false;//跟踪失败
		      nextRelocIdx = -1;

		      unmappedTrackedFramesMutex.lock();
		      unmappedTrackedFramesSignal.notify_one();
		      unmappedTrackedFramesMutex.unlock();

		      manualTrackingLossIndicated = false;
		      return;
	      }
            // 打印跟踪信息
	      if(plotTracking)
	      {
		      Eigen::Matrix<float, 20, 1> data;
		      data.setZero();
		      data[0] = tracker->lastResidual;// 光度误差

		      data[3] = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);// 好的匹配点占总的匹配点比例
		      // 可以理解为当前帧和参考帧重叠区域的比例。
		      data[4] = 4*tracker->lastGoodCount / (width*height); // 好的匹配点占总的像素点比例
		      data[5] = tracker->pointUsage;// 深度改变均值

		      data[6] = tracker->affineEstimation_a;// 参考帧灰度值仿射变换系数a
		      data[7] = tracker->affineEstimation_b;// 仿射变换系数a
		      outputWrapper->publishDebugInfo(data);// 打印信息
	      }
	      // 添加当前帧到帧地图
	      keyFrameGraph->addFrame(trackingNewFrame.get());
	      //Sim3 lastTrackedCamToWorld = mostCurrentTrackedFrame->getScaledCamToWorld();//  mostCurrentTrackedFrame->TrackingParent->getScaledCamToWorld() * sim3FromSE3(mostCurrentTrackedFrame->thisToParent_SE3TrackingResult, 1.0);
	      if (outputWrapper != 0)
	      {
		      outputWrapper->publishTrackedFrame(trackingNewFrame.get());
	      }

// 第六步.  关键帧筛选：通过计算得分确定是否构造新的关键帧
            // 根据运动距离来确定，如果当前相机运动 距离参考帧(最近的一个关键帧) 过远 则把 当前帧创建为关键帧。
            // 并且给出了距离函数 d =ξ转置 * W * ξ
            // 其中W是权重矩阵。并且距离阈值根据当前帧场景平均逆深度来确定。
	      // Keyframe selection
	      latestTrackedFrame = trackingNewFrame;// 上一次跟踪的帧
	      // 条件是当前建图线程是否已经更新好上一帧关键帧以及跟踪最近的关键帧（参考帧）的图像帧个数不小于一定值（MIN_NUM_MAPPED，为5）。
	      if (!my_createNewKeyframe && currentKeyFrame->numMappedOnThisTotal > MIN_NUM_MAPPED)
	      {
		// 如果满足上述的条件则计算得分，得分大于一定阈值则确定构建新的关键帧。
		// 位移向量先乘以一个平均场景逆深度：
		// 显然，大场景的逆深度会小，小场景的逆深度会大，这相当于一个权重，对于相同的位移，认为小场景的运动程度比较大。
		      Sophus::Vector3d dist = newRefToFrame_poseUpdate.translation() * currentKeyFrame->meanIdepth;
	        // 阈值变量：  
		      float minVal = fmin(0.2f + keyFrameGraph->keyframesAll.size() * 0.8f / INITIALIZATION_PHASE_COUNT, 1.0f);
               // 这里的两个操作都是使得初始化阶段当 关键帧比较少的时候（INITIALIZATION_PHASE_COUNT为5）放宽了阈值，之后就是１。
		      if(keyFrameGraph->keyframesAll.size() < INITIALIZATION_PHASE_COUNT)	minVal *= 0.7;//这里可以调整　以适应初始化时　关键帧较少
              // 这里的得分和当前帧和参考帧之间的位移大小有关。参数有加权距离平方，以及当前帧和参考帧重叠区域的比例
		      lastTrackingClosenessScore = trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage);
              // 得分大于阈值的化就可以　将当当前帧　创建为关键帧
		      if (lastTrackingClosenessScore > minVal)
		      {
			      createNewKeyFrame = true;

			      if(enablePrintDebugInfo && printKeyframeSelectionInfo)
				      printf("SELECT %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));
		      }
		      else
		      {
			      if(enablePrintDebugInfo && printKeyframeSelectionInfo)
				      printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));

		      }
	      }


	      unmappedTrackedFramesMutex.lock();
	      // 是用来保存图像帧的队列 最近50帧
	      if(unmappedTrackedFrames.size() < 50 || (unmappedTrackedFrames.size() < 100 && trackingNewFrame->getTrackingParent()->numMappedOnThisTotal < 10))
		      unmappedTrackedFrames.push_back(trackingNewFrame);
	      unmappedTrackedFramesSignal.notify_one();
	      unmappedTrackedFramesMutex.unlock();

	      // implement blocking
	      if(blockUntilMapped && trackingIsGood)
	      {
		      boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
		      while(unmappedTrackedFrames.size() > 0)
		      {
			      //printf("TRACKING IS BLOCKING, waiting for %d frames to finish mapping.\n", (int)unmappedTrackedFrames.size());
			      newFrameMappedSignal.wait(lock);
		      }
		      lock.unlock();
	      }
      }


      float SlamSystem::tryTrackSim3(
		      TrackingReference* A, TrackingReference* B,
		      int lvlStart, int lvlEnd,
		      bool useSSE,
		      Sim3 &AtoB, Sim3 &BtoA,
		      KFConstraintStruct* e1, KFConstraintStruct* e2 )
      {
	      BtoA = constraintTracker->trackFrameSim3(
			      A,
			      B->keyframe,
			      BtoA,
			      lvlStart,lvlEnd);
	      Matrix7x7 BtoAInfo = constraintTracker->lastSim3Hessian;
	      float BtoA_meanResidual = constraintTracker->lastResidual;
	      float BtoA_meanDResidual = constraintTracker->lastDepthResidual;
	      float BtoA_meanPResidual = constraintTracker->lastPhotometricResidual;
	      float BtoA_usage = constraintTracker->pointUsage;


	      if (constraintTracker->diverged ||
		      BtoA.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon() ||
		      BtoA.scale() < Sophus::SophusConstants<sophusType>::epsilon() ||
		      BtoAInfo(0,0) == 0 ||
		      BtoAInfo(6,6) == 0)
	      {
		      return 1e20;
	      }


	      AtoB = constraintTracker->trackFrameSim3(
			      B,
			      A->keyframe,
			      AtoB,
			      lvlStart,lvlEnd);
	      Matrix7x7 AtoBInfo = constraintTracker->lastSim3Hessian;
	      float AtoB_meanResidual = constraintTracker->lastResidual;
	      float AtoB_meanDResidual = constraintTracker->lastDepthResidual;
	      float AtoB_meanPResidual = constraintTracker->lastPhotometricResidual;
	      float AtoB_usage = constraintTracker->pointUsage;


	      if (constraintTracker->diverged ||
		      AtoB.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon() ||
		      AtoB.scale() < Sophus::SophusConstants<sophusType>::epsilon() ||
		      AtoBInfo(0,0) == 0 ||
		      AtoBInfo(6,6) == 0)
	      {
		      return 1e20;
	      }

	      // Propagate uncertainty (with d(a * b) / d(b) = Adj_a) and calculate Mahalanobis norm
	      Matrix7x7 datimesb_db = AtoB.cast<float>().Adj();
	      Matrix7x7 diffHesse = (AtoBInfo.inverse() + datimesb_db * BtoAInfo.inverse() * datimesb_db.transpose()).inverse();
	      Vector7 diff = (AtoB * BtoA).log().cast<float>();


	      float reciprocalConsistency = (diffHesse * diff).dot(diff);


	      if(e1 != 0 && e2 != 0)
	      {
		      e1->firstFrame = A->keyframe;
		      e1->secondFrame = B->keyframe;
		      e1->secondToFirst = BtoA;
		      e1->information = BtoAInfo.cast<double>();
		      e1->meanResidual = BtoA_meanResidual;
		      e1->meanResidualD = BtoA_meanDResidual;
		      e1->meanResidualP = BtoA_meanPResidual;
		      e1->usage = BtoA_usage;

		      e2->firstFrame = B->keyframe;
		      e2->secondFrame = A->keyframe;
		      e2->secondToFirst = AtoB;
		      e2->information = AtoBInfo.cast<double>();
		      e2->meanResidual = AtoB_meanResidual;
		      e2->meanResidualD = AtoB_meanDResidual;
		      e2->meanResidualP = AtoB_meanPResidual;
		      e2->usage = AtoB_usage;

		      e1->reciprocalConsistency = e2->reciprocalConsistency = reciprocalConsistency;
	      }

	      return reciprocalConsistency;
      }


      void SlamSystem::testConstraint(
		      Frame* candidate,
		      KFConstraintStruct* &e1_out, KFConstraintStruct* &e2_out,
		      Sim3 candidateToFrame_initialEstimate,
		      float strictness)
      {
	      candidateTrackingReference->importFrame(candidate);

	      Sim3 FtoC = candidateToFrame_initialEstimate.inverse(), CtoF = candidateToFrame_initialEstimate;
	      Matrix7x7 FtoCInfo, CtoFInfo;

	      float err_level3 = tryTrackSim3(
			      newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			      SIM3TRACKING_MAX_LEVEL-1, 3,
			      USESSE,
			      FtoC, CtoF);

	      if(err_level3 > 3000*strictness)
	      {
		      if(enablePrintDebugInfo && printConstraintSearchInfo)
			      printf("FAILE %d -> %d (lvl %d): errs (%.1f / - / -).",
				      newKFTrackingReference->frameID, candidateTrackingReference->frameID,
				      3,
				      sqrtf(err_level3));

		      e1_out = e2_out = 0;

		      newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		      return;
	      }

	      float err_level2 = tryTrackSim3(
			      newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			      2, 2,
			      USESSE,
			      FtoC, CtoF);

	      if(err_level2 > 4000*strictness)
	      {
		      if(enablePrintDebugInfo && printConstraintSearchInfo)
			      printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / -).",
				      newKFTrackingReference->frameID, candidateTrackingReference->frameID,
				      2,
				      sqrtf(err_level3), sqrtf(err_level2));

		      e1_out = e2_out = 0;
		      newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		      return;
	      }

	      e1_out = new KFConstraintStruct();
	      e2_out = new KFConstraintStruct();


	      float err_level1 = tryTrackSim3(
			      newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			      1, 1,
			      USESSE,
			      FtoC, CtoF, e1_out, e2_out);

	      if(err_level1 > 6000*strictness)
	      {
		      if(enablePrintDebugInfo && printConstraintSearchInfo)
			      printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / %.1f).",
					      newKFTrackingReference->frameID, candidateTrackingReference->frameID,
					      1,
					      sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));

		      delete e1_out;
		      delete e2_out;
		      e1_out = e2_out = 0;
		      newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		      return;
	      }


	      if(enablePrintDebugInfo && printConstraintSearchInfo)
		      printf("ADDED %d -> %d: errs (%.1f / %.1f / %.1f).",
			      newKFTrackingReference->frameID, candidateTrackingReference->frameID,
			      sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));


	      const float kernelDelta = 5 * sqrt(6000*loopclosureStrictness);
	      e1_out->robustKernel = new g2o::RobustKernelHuber();
	      e1_out->robustKernel->setDelta(kernelDelta);
	      e2_out->robustKernel = new g2o::RobustKernelHuber();
	      e2_out->robustKernel->setDelta(kernelDelta);
      }
// 测试闭环 findConstraintsForNewKeyFrames
// 该函数主要是根据视差、关键帧连接关系，找出并且在删选处候选帧，然后对每个候选帧和测试的关键帧之间进行双向sim3跟踪，
// 如果求解出的两个李代数满足马氏距离在一定范围内，则认为是闭环成功，并且在位姿图中添加边的约束。
      int SlamSystem::findConstraintsForNewKeyFrames(Frame* newKeyFrame, bool forceParent, bool useFABMAP, float closeCandidatesTH)
      {
	      if(!newKeyFrame->hasTrackingParent())
	      {
		      newConstraintMutex.lock();
		      keyFrameGraph->addKeyFrame(newKeyFrame);
		      newConstraintAdded = true;
		      newConstraintCreatedSignal.notify_all();
		      newConstraintMutex.unlock();
		      return 0;
	      }

	      if(!forceParent && (newKeyFrame->lastConstraintTrackedCamToWorld * newKeyFrame->getScaledCamToWorld().inverse()).log().norm() < 0.01)
		      return 0;


	      newKeyFrame->lastConstraintTrackedCamToWorld = newKeyFrame->getScaledCamToWorld();

	      // =============== get all potential candidates and their initial relative pose. =================
	      std::vector<KFConstraintStruct*, Eigen::aligned_allocator<KFConstraintStruct*> > constraints;
	      Frame* fabMapResult = 0;
	      std::unordered_set<Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
		      Eigen::aligned_allocator< Frame* > > candidates = trackableKeyFrameSearch->findCandidates(newKeyFrame, fabMapResult, useFABMAP, closeCandidatesTH);
	      std::map< Frame*, Sim3, std::less<Frame*>, Eigen::aligned_allocator<std::pair<Frame*, Sim3> > > candidateToFrame_initialEstimateMap;


	      // erase the ones that are already neighbours.
	      for(std::unordered_set<Frame*>::iterator c = candidates.begin(); c != candidates.end();)
	      {
		      if(newKeyFrame->neighbors.find(*c) != newKeyFrame->neighbors.end())
		      {
			      if(enablePrintDebugInfo && printConstraintSearchInfo)
				      printf("SKIPPING %d on %d cause it already exists as constraint.\n", (*c)->id(), newKeyFrame->id());
			      c = candidates.erase(c);
		      }
		      else
			      ++c;
	      }

	      poseConsistencyMutex.lock_shared();
	      for (Frame* candidate : candidates)
	      {
		      Sim3 candidateToFrame_initialEstimate = newKeyFrame->getScaledCamToWorld().inverse() * candidate->getScaledCamToWorld();
		      candidateToFrame_initialEstimateMap[candidate] = candidateToFrame_initialEstimate;
	      }

	      std::unordered_map<Frame*, int> distancesToNewKeyFrame;
	      if(newKeyFrame->hasTrackingParent())
		      keyFrameGraph->calculateGraphDistancesToFrame(newKeyFrame->getTrackingParent(), &distancesToNewKeyFrame);
	      poseConsistencyMutex.unlock_shared();


	      // =============== distinguish between close and "far" candidates in Graph =================
	      // Do a first check on trackability of close candidates.
	      std::unordered_set<Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
		      Eigen::aligned_allocator< Frame* > > closeCandidates;
	      std::vector<Frame*, Eigen::aligned_allocator<Frame*> > farCandidates;
	      Frame* parent = newKeyFrame->hasTrackingParent() ? newKeyFrame->getTrackingParent() : 0;

	      int closeFailed = 0;
	      int closeInconsistent = 0;

	      SO3 disturbance = SO3::exp(Sophus::Vector3d(0.05,0,0));

	      for (Frame* candidate : candidates)
	      {
		      if (candidate->id() == newKeyFrame->id())
			      continue;
		      if(!candidate->pose->isInGraph)
			      continue;
		      if(newKeyFrame->hasTrackingParent() && candidate == newKeyFrame->getTrackingParent())
			      continue;
		      if(candidate->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
			      continue;

		      SE3 c2f_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate].inverse()).inverse();
		      c2f_init.so3() = c2f_init.so3() * disturbance;
		      SE3 c2f = constraintSE3Tracker->trackFrameOnPermaref(candidate, newKeyFrame, c2f_init);
		      if(!constraintSE3Tracker->trackingWasGood) {closeFailed++; continue;}


		      SE3 f2c_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate]).inverse();
		      f2c_init.so3() = disturbance * f2c_init.so3();
		      SE3 f2c = constraintSE3Tracker->trackFrameOnPermaref(newKeyFrame, candidate, f2c_init);
		      if(!constraintSE3Tracker->trackingWasGood) {closeFailed++; continue;}

		      if((f2c.so3() * c2f.so3()).log().norm() >= 0.09) {closeInconsistent++; continue;}

		      closeCandidates.insert(candidate);
	      }



	      int farFailed = 0;
	      int farInconsistent = 0;
	      for (Frame* candidate : candidates)
	      {
		      if (candidate->id() == newKeyFrame->id())
			      continue;
		      if(!candidate->pose->isInGraph)
			      continue;
		      if(newKeyFrame->hasTrackingParent() && candidate == newKeyFrame->getTrackingParent())
			      continue;
		      if(candidate->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
			      continue;

		      if(candidate == fabMapResult)
		      {
			      farCandidates.push_back(candidate);
			      continue;
		      }

		      if(distancesToNewKeyFrame.at(candidate) < 4)
			      continue;

		      farCandidates.push_back(candidate);
	      }




	      int closeAll = closeCandidates.size();
	      int farAll = farCandidates.size();

	      // erase the ones that we tried already before (close)
	      for(std::unordered_set<Frame*>::iterator c = closeCandidates.begin(); c != closeCandidates.end();)
	      {
		      if(newKeyFrame->trackingFailed.find(*c) == newKeyFrame->trackingFailed.end())
		      {
			      ++c;
			      continue;
		      }
		      auto range = newKeyFrame->trackingFailed.equal_range(*c);

		      bool skip = false;
		      Sim3 f2c = candidateToFrame_initialEstimateMap[*c].inverse();
		      for (auto it = range.first; it != range.second; ++it)
		      {
			      if((f2c * it->second).log().norm() < 0.1)
			      {
				      skip=true;
				      break;
			      }
		      }

		      if(skip)
		      {
			      if(enablePrintDebugInfo && printConstraintSearchInfo)
				      printf("SKIPPING %d on %d (NEAR), cause we already have tried it.\n", (*c)->id(), newKeyFrame->id());
			      c = closeCandidates.erase(c);
		      }
		      else
			      ++c;
	      }

	      // erase the ones that are already neighbours (far)
	      for(unsigned int i=0;i<farCandidates.size();i++)
	      {
		      if(newKeyFrame->trackingFailed.find(farCandidates[i]) == newKeyFrame->trackingFailed.end())
			      continue;

		      auto range = newKeyFrame->trackingFailed.equal_range(farCandidates[i]);

		      bool skip = false;
		      for (auto it = range.first; it != range.second; ++it)
		      {
			      if((it->second).log().norm() < 0.2)
			      {
				      skip=true;
				      break;
			      }
		      }

		      if(skip)
		      {
			      if(enablePrintDebugInfo && printConstraintSearchInfo)
				      printf("SKIPPING %d on %d (FAR), cause we already have tried it.\n", farCandidates[i]->id(), newKeyFrame->id());
			      farCandidates[i] = farCandidates.back();
			      farCandidates.pop_back();
			      i--;
		      }
	      }



	      if (enablePrintDebugInfo && printConstraintSearchInfo)
		      printf("Final Loop-Closure Candidates: %d / %d close (%d failed, %d inconsistent) + %d / %d far (%d failed, %d inconsistent) = %d\n",
				      (int)closeCandidates.size(),closeAll, closeFailed, closeInconsistent,
				      (int)farCandidates.size(), farAll, farFailed, farInconsistent,
				      (int)closeCandidates.size() + (int)farCandidates.size());



	      // =============== limit number of close candidates ===============
	      // while too many, remove the one with the highest connectivity.
	      while((int)closeCandidates.size() > maxLoopClosureCandidates)
	      {
		      Frame* worst = 0;
		      int worstNeighbours = 0;
		      for(Frame* f : closeCandidates)
		      {
			      int neightboursInCandidates = 0;
			      for(Frame* n : f->neighbors)
				      if(closeCandidates.find(n) != closeCandidates.end())
					      neightboursInCandidates++;

			      if(neightboursInCandidates > worstNeighbours || worst == 0)
			      {
				      worst = f;
				      worstNeighbours = neightboursInCandidates;
			      }
		      }

		      closeCandidates.erase(worst);
	      }


	      // =============== limit number of far candidates ===============
	      // delete randomly
	      int maxNumFarCandidates = (maxLoopClosureCandidates +1) / 2;
	      if(maxNumFarCandidates < 5) maxNumFarCandidates = 5;
	      while((int)farCandidates.size() > maxNumFarCandidates)
	      {
		      int toDelete = rand() % farCandidates.size();
		      if(farCandidates[toDelete] != fabMapResult)
		      {
			      farCandidates[toDelete] = farCandidates.back();
			      farCandidates.pop_back();
		      }
	      }







	      // =============== TRACK! ===============

	      // make tracking reference for newKeyFrame.
	      newKFTrackingReference->importFrame(newKeyFrame);


	      for (Frame* candidate : closeCandidates)
	      {
		      KFConstraintStruct* e1=0;
		      KFConstraintStruct* e2=0;

		      testConstraint(
				      candidate, e1, e2,
				      candidateToFrame_initialEstimateMap[candidate],
				      loopclosureStrictness);

		      if(enablePrintDebugInfo && printConstraintSearchInfo)
			      printf(" CLOSE (%d)\n", distancesToNewKeyFrame.at(candidate));

		      if(e1 != 0)
		      {
			      constraints.push_back(e1);
			      constraints.push_back(e2);

			      // delete from far candidates if it's in there.
			      for(unsigned int k=0;k<farCandidates.size();k++)
			      {
				      if(farCandidates[k] == candidate)
				      {
					      if(enablePrintDebugInfo && printConstraintSearchInfo)
						      printf(" DELETED %d from far, as close was successful!\n", candidate->id());

					      farCandidates[k] = farCandidates.back();
					      farCandidates.pop_back();
				      }
			      }
		      }
	      }


	      for (Frame* candidate : farCandidates)
	      {
		      KFConstraintStruct* e1=0;
		      KFConstraintStruct* e2=0;

		      testConstraint(
				      candidate, e1, e2,
				      Sim3(),
				      loopclosureStrictness);

		      if(enablePrintDebugInfo && printConstraintSearchInfo)
			      printf(" FAR (%d)\n", distancesToNewKeyFrame.at(candidate));

		      if(e1 != 0)
		      {
			      constraints.push_back(e1);
			      constraints.push_back(e2);
		      }
	      }



	      if(parent != 0 && forceParent)
	      {
		      KFConstraintStruct* e1=0;
		      KFConstraintStruct* e2=0;
		      testConstraint(
				      parent, e1, e2,
				      candidateToFrame_initialEstimateMap[parent],
				      100);
		      if(enablePrintDebugInfo && printConstraintSearchInfo)
			      printf(" PARENT (0)\n");

		      if(e1 != 0)
		      {
			      constraints.push_back(e1);
			      constraints.push_back(e2);
		      }
		      else
		      {
			      float downweightFac = 5;
			      const float kernelDelta = 5 * sqrt(6000*loopclosureStrictness) / downweightFac;
			      printf("warning: reciprocal tracking on new frame failed badly, added odometry edge (Hacky).\n");

			      poseConsistencyMutex.lock_shared();
			      constraints.push_back(new KFConstraintStruct());
			      constraints.back()->firstFrame = newKeyFrame;
			      constraints.back()->secondFrame = newKeyFrame->getTrackingParent();
			      constraints.back()->secondToFirst = constraints.back()->firstFrame->getScaledCamToWorld().inverse() * constraints.back()->secondFrame->getScaledCamToWorld();
			      constraints.back()->information  <<
					      0.8098,-0.1507,-0.0557, 0.1211, 0.7657, 0.0120, 0,
					      -0.1507, 2.1724,-0.1103,-1.9279,-0.1182, 0.1943, 0,
					      -0.0557,-0.1103, 0.2643,-0.0021,-0.0657,-0.0028, 0.0304,
					      0.1211,-1.9279,-0.0021, 2.3110, 0.1039,-0.0934, 0.0005,
					      0.7657,-0.1182,-0.0657, 0.1039, 1.0545, 0.0743,-0.0028,
					      0.0120, 0.1943,-0.0028,-0.0934, 0.0743, 0.4511, 0,
					      0,0, 0.0304, 0.0005,-0.0028, 0, 0.0228;
			      constraints.back()->information *= (1e9/(downweightFac*downweightFac));

			      constraints.back()->robustKernel = new g2o::RobustKernelHuber();
			      constraints.back()->robustKernel->setDelta(kernelDelta);

			      constraints.back()->meanResidual = 10;
			      constraints.back()->meanResidualD = 10;
			      constraints.back()->meanResidualP = 10;
			      constraints.back()->usage = 0;

			      poseConsistencyMutex.unlock_shared();
		      }
	      }


	      newConstraintMutex.lock();

	      keyFrameGraph->addKeyFrame(newKeyFrame);
	      for(unsigned int i=0;i<constraints.size();i++)
		      keyFrameGraph->insertConstraint(constraints[i]);


	      newConstraintAdded = true;
	      newConstraintCreatedSignal.notify_all();
	      newConstraintMutex.unlock();

	      newKFTrackingReference->invalidate();
	      candidateTrackingReference->invalidate();



	      return constraints.size();
      }




      bool SlamSystem::optimizationIteration(int itsPerTry, float minChange)
      {
	      struct timeval tv_start, tv_end;
	      gettimeofday(&tv_start, NULL);



	      g2oGraphAccessMutex.lock();

	      // lock new elements buffer & take them over.
	      newConstraintMutex.lock();
	      keyFrameGraph->addElementsFromBuffer();
	      newConstraintMutex.unlock();


	      // Do the optimization. This can take quite some time!
	      int its = keyFrameGraph->optimize(itsPerTry);
	      

	      // save the optimization result.
	      poseConsistencyMutex.lock_shared();
	      keyFrameGraph->keyframesAllMutex.lock_shared();
	      float maxChange = 0;
	      float sumChange = 0;
	      float sum = 0;
	      for(size_t i=0;i<keyFrameGraph->keyframesAll.size(); i++)
	      {
		      // set edge error sum to zero
		      keyFrameGraph->keyframesAll[i]->edgeErrorSum = 0;
		      keyFrameGraph->keyframesAll[i]->edgesNum = 0;

		      if(!keyFrameGraph->keyframesAll[i]->pose->isInGraph) continue;



		      // get change from last optimization
		      Sim3 a = keyFrameGraph->keyframesAll[i]->pose->graphVertex->estimate();
		      Sim3 b = keyFrameGraph->keyframesAll[i]->getScaledCamToWorld();
		      Sophus::Vector7f diff = (a*b.inverse()).log().cast<float>();


		      for(int j=0;j<7;j++)
		      {
			      float d = fabsf((float)(diff[j]));
			      if(d > maxChange) maxChange = d;
			      sumChange += d;
		      }
		      sum +=7;

		      // set change
		      keyFrameGraph->keyframesAll[i]->pose->setPoseGraphOptResult(
				      keyFrameGraph->keyframesAll[i]->pose->graphVertex->estimate());

		      // add error
		      for(auto edge : keyFrameGraph->keyframesAll[i]->pose->graphVertex->edges())
		      {
			      keyFrameGraph->keyframesAll[i]->edgeErrorSum += ((EdgeSim3*)(edge))->chi2();
			      keyFrameGraph->keyframesAll[i]->edgesNum++;
		      }
	      }

	      haveUnmergedOptimizationOffset = true;
	      keyFrameGraph->keyframesAllMutex.unlock_shared();
	      poseConsistencyMutex.unlock_shared();

	      g2oGraphAccessMutex.unlock();

	      if(enablePrintDebugInfo && printOptimizationInfo)
		      printf("did %d optimization iterations. Max Pose Parameter Change: %f; avgChange: %f. %s\n", its, maxChange, sumChange / sum,
				      maxChange > minChange && its == itsPerTry ? "continue optimizing":"Waiting for addition to graph.");


	      gettimeofday(&tv_end, NULL);
	      msOptimizationIteration = 0.9*msOptimizationIteration + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	      nOptimizationIteration++;


	      return maxChange > minChange && its == itsPerTry;
      }

      void SlamSystem::optimizeGraph()
      {
	      boost::unique_lock<boost::mutex> g2oLock(g2oGraphAccessMutex);
	      keyFrameGraph->optimize(1000);
	      g2oLock.unlock();
	      mergeOptimizationOffset();
      }


      SE3 SlamSystem::getCurrentPoseEstimate()
      {
	      SE3 camToWorld = SE3();
	      keyFrameGraph->allFramePosesMutex.lock_shared();
	      if(keyFrameGraph->allFramePoses.size() > 0)
		      camToWorld = se3FromSim3(keyFrameGraph->allFramePoses.back()->getCamToWorld());
	      keyFrameGraph->allFramePosesMutex.unlock_shared();
	      return camToWorld;
      }

      std::vector<FramePoseStruct*, Eigen::aligned_allocator<FramePoseStruct*> > SlamSystem::getAllPoses()
      {
	      return keyFrameGraph->allFramePoses;
      }
