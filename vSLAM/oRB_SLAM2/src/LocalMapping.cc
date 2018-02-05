/**
* This file is part of ORB-SLAM2.
* 
* LocalMapping作用是将Tracking中送来的关键帧放在mlNewKeyFrame列表中；
* 处理新关键帧，地图点检查剔除，生成新地图点，Local BA，关键帧剔除。
* 主要工作在于维护局部地图，也就是SLAM中的Mapping。
* 
* 
* Tracking线程 只是判断当前帧是否需要加入关键帧，并没有真的加入地图，
* 因为Tracking线程的主要功能是局部定位，
* 
* 而处理地图中的关键帧，地图点，包括如何加入，
* 如何删除的工作是在LocalMapping线程完成的
* 
* 建图 
* 处理新的关键帧 KeyFrame 完成局部地图构建
* 插入关键帧 ------>  处理地图点(筛选生成的地图点 生成地图点)  -------->  局部 BA最小化重投影误差  -调整-------->   筛选 新插入的 关键帧
*
* mlNewKeyFrames     list 列表队列存储关键帧
* 1】检查队列
*       CheckNewKeyFrames();
* 
* 2】处理新关键帧 Proces New Key Frames 
* 	ProcessNewKeyFrame(); 更新地图点MapPoints 和 关键帧 KepFrame 的关联关系  UpdateConnections() 更新关联关系
* 
* 3】剔除 地图点 MapPoints
*       删除地图中新添加的但 质量不好的 地图点
*       a)  IncreaseFound / IncreaseVisible < 25%
*       b) 观测到该 点的关键帧太少
* 
* 4】生成 地图点 MapPoints
* 	运动过程中和共视程度比较高的 关键帧 通过三角化 恢复出的一些地图点 
* 
* 5】地图点融合 MapPoints
*       检测当前关键帧和相邻 关键帧(两级相邻) 中 重复的 地图点 
* 
* 6】局部 BA 最小化重投影误差
*      和当前关键帧相邻的关键帧 中相匹配的 地图点对 最局部 BA最小化重投影误差优化点坐标 和 位姿
* 
* 7】关键帧剔除
*      其90%以上的 地图点 能够被其他 共视 关键帧(至少3个) 观测到，认为该关键帧多余，可以删除
* 
*/




#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

	LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
	    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
	    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
	{
	}

	void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
	{
	    mpLoopCloser = pLoopCloser;
	}

	void LocalMapping::SetTracker(Tracking *pTracker)
	{
	    mpTracker=pTracker;
	}

	void LocalMapping::Run()
	{

	    mbFinished = false;

	    while(1)
	    {
		// Tracking will see that Local Mapping is busy
// 步骤1：设置进程间的访问标志 告诉Tracking线程，LocalMapping线程正在处理新的关键帧，处于繁忙状态
               // LocalMapping线程处理的关键帧都是Tracking线程发过的
               // 在LocalMapping线程还没有处理完关键帧之前Tracking线程最好不要发送太快
		SetAcceptKeyFrames(false);

		// Check if there are keyframes in the queue
	// 等待处理的关键帧列表不为空
		if(CheckNewKeyFrames())
		{
		    // BoW conversion and insertion in Map
// 步骤2：计算关键帧特征点的词典单词向量BoW映射，将关键帧插入地图
		    ProcessNewKeyFrame();

		    // Check recent MapPoints
		  // 剔除ProcessNewKeyFrame函数中引入的不合格MapPoints
// 步骤3：对新添加的地图点融合 对于 ProcessNewKeyFrame 和 CreateNewMapPoints 中 最近添加的MapPoints进行检查剔除	    
		  //   MapPointCulling();

		    // Triangulate new MapPoints
		    
// 步骤4： 创建新的地图点 相机运动过程中与相邻关键帧通过三角化恢复出一些新的地图点MapPoints	    
		    CreateNewMapPoints();
		    
		    MapPointCulling();// 从上面 移到下面

	      // 已经处理完队列中的最后的一个关键帧
		    if(!CheckNewKeyFrames())
		    {
			// Find more matches in neighbor keyframes and fuse point duplications
// 步骤5：相邻帧地图点融合 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
			SearchInNeighbors();
		    }

		    mbAbortBA = false;

		// 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
		    if(!CheckNewKeyFrames() && !stopRequested())
		    {
// 步骤6：局部地图优化 Local BA
			if(mpMap->KeyFramesInMap() > 2)
			    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

// 步骤7： 关键帧融合 检测并剔除当前帧相邻的关键帧中冗余的关键帧 Check redundant local Keyframes
	                // 剔除的标准是：该关键帧的90%的MapPoints可以被其它关键帧观测到		
			// Tracking中先把关键帧交给LocalMapping线程
			 // 并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密
                        // 在这里再删除冗余的关键帧
			KeyFrameCulling();
		    }
// 步骤8：将当前帧加入到闭环检测队列中
		    mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
		}
// 步骤9：等待线程空闲 完成一帧关键帧的插入融合工作
		else if(Stop())
		{
		    // Safe area to stop
		    while(isStopped() && !CheckFinish())
		    {
			usleep(3000);
		    }
		    if(CheckFinish())//检查 是否完成
			break;
		}
		
               // 检查重置
		ResetIfRequested();

		// Tracking will see that Local Mapping is not busy
// 步骤10：告诉 	Tracking 线程  Local Mapping 线程 空闲 可一处理接收 下一个 关键帧
		SetAcceptKeyFrames(true);

		if(CheckFinish())
		    break;

		usleep(3000);
	    }

	    SetFinish();
	}

/**
 * @brief 插入关键帧
 *
 * 将关键帧插入到地图中，以便将来进行局部地图优化
 * 这里仅仅是将关键帧插入到列表中进行等待
 * @param pKF KeyFrame
 */
	void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
	{
	    unique_lock<mutex> lock(mMutexNewKFs);
	     // 将关键帧插入到 等待处理的关键帧列表中
	    mlNewKeyFrames.push_back(pKF);
	    mbAbortBA=true;// BA优化停止
	}

/**
 * @brief    查看列表中是否有等待 被处理的关键帧
 * @return 如果存在，返回true
 */
	bool LocalMapping::CheckNewKeyFrames()
	{
	    unique_lock<mutex> lock(mMutexNewKFs);
	    return(!mlNewKeyFrames.empty());// 等待处理的关键帧列表是否为空
	}


  /*
  a. 根据词典 计算当前关键帧Bow，便于后面三角化恢复新地图点；
  b. 将TrackLocalMap中跟踪局部地图匹配上的地图点绑定到当前关键帧
      （在Tracking线程中只是通过匹配进行局部地图跟踪，优化当前关键帧姿态），
      也就是在graph 中加入当前关键帧作为node，并更新edge。
      
      而CreateNewMapPoint()中则通过当前关键帧，在局部地图中添加与新的地图点；

  c. 更新加入当前关键帧之后关键帧之间的连接关系，包括更新Covisibility图和Essential图
  （最小生成树spanning tree，共视关系好的边subset of edges from covisibility graph 
    with high covisibility (θ=100)， 闭环边）。
    */
  
  /**
 * @brief 处理列表中的关键帧
 * 
 * - 计算Bow，加速三角化新的MapPoints
 * - 关联当前关键帧至MapPoints，并更新MapPoints的平均观测方向和观测距离范围
 * - 插入关键帧，更新Covisibility图和Essential图
 * @see VI-A keyframe insertion
 */
	void LocalMapping::ProcessNewKeyFrame()
	{	  
// 步骤1：从缓冲队列中取出一帧待处理的关键帧
             // Tracking线程向LocalMapping中插入关键帧存在该队列中
	    {
		unique_lock<mutex> lock(mMutexNewKFs);
		// 从列表中获得一个等待被插入的关键帧
		mpCurrentKeyFrame = mlNewKeyFrames.front();
		mlNewKeyFrames.pop_front();// 出队
	    }

	    // Compute Bags of Words structures
// 步骤2：计算该关键帧特征点的Bow映射关系    
	    //  根据词典 计算当前关键帧Bow，便于后面三角化恢复新地图点    
	    mpCurrentKeyFrame->ComputeBoW();// 帧描述子 用字典单词线性表示的 向量

	    
	    // Associate MapPoints to the new keyframe and update normal and descriptor
	    // 当前关键帧  TrackLocalMap中跟踪局部地图 匹配上的 地图点
// 步骤3：跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定
	      // 在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配，
	      // 但没有对这些匹配上的MapPoints与当前帧进行关联    
	    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
	    for(size_t i=0; i<vpMapPointMatches.size(); i++)
	    {
		MapPoint* pMP = vpMapPointMatches[i];// 每一个与当前关键帧匹配好的地图点
		if(pMP)//地图点存在
		{
		    if(!pMP->isBad())
		    {
		       // 为当前帧在tracking过重跟踪到的MapPoints更新属性
			if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))//下视野内
			{
			    pMP->AddObservation(mpCurrentKeyFrame, i);// 地图点添加关键帧
			    pMP->UpdateNormalAndDepth();// 地图点更新 平均观测方向 和 观测距离深度
			    pMP->ComputeDistinctiveDescriptors();// 加入关键帧后，更新地图点的最佳描述子
			}
			else // 双目追踪时插入的点 可能不在 帧上this can only happen for new stereo points inserted by the Tracking
			{
			   // 将双目或RGBD跟踪过程中新插入的MapPoints放入mlpRecentAddedMapPoints，等待检查
                           // CreateNewMapPoints函数中通过三角化也会生成MapPoints
                           // 这些MapPoints都会经过MapPointCulling函数的检验
			    mlpRecentAddedMapPoints.push_back(pMP);
			    // 候选待检查地图点存放在mlpRecentAddedMapPoints
			}
		    }
		}
	    }    

	    // Update links in the Covisibility Graph
// 步骤4：更新关键帧间的连接关系，Covisibility图和Essential图(tree)
	    mpCurrentKeyFrame->UpdateConnections();

	    // Insert Keyframe in Map
// 步骤5：将该关键帧插入到地图中
	    mpMap->AddKeyFrame(mpCurrentKeyFrame);
	}
	
	
/**
 * @brief 剔除ProcessNewKeyFrame(不在帧上的地图点 进入待查list)和
 *             CreateNewMapPoints(两帧三角变换产生的新地图点进入 待查list)
 *             函数中引入的质量不好的MapPoints
 * @see VI-B recent map points culling
 */	
// 对于ProcessNewKeyFrame和CreateNewMapPoints中最近添加的MapPoints进行检查剔除
	void LocalMapping::MapPointCulling()
	{
	    // Check Recent Added MapPoints
	    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();//待检测的地图点 迭代器
	    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;//当前关键帧id

	    //  从添加该地图点的关键帧算起的 初始三个关键帧，
	    // 第一帧不算，后面两帧看到该地图点的帧数，对于单目<=2，对于双目和RGBD<=3；
	    int nThObs;
	    if(mbMonocular)
		nThObs = 2;
	    else
		nThObs = 3;
	    const int cnThObs = nThObs;
 // 遍历等待检查的地图点MapPoints
	    while(lit !=mlpRecentAddedMapPoints.end())
	    {
		MapPoint* pMP = *lit;//  新添加的地图点
		if(pMP->isBad())
		{
 // 步骤1：已经是坏点的MapPoints直接从检查链表中删除	  
		    lit = mlpRecentAddedMapPoints.erase(lit);
		}
		//  跟踪（匹配上）到该地图点的普通帧帧数（IncreaseFound）< 应该观测到该地图点的普通帧数量（25%*IncreaseVisible）：
		// 该地图点虽在视野范围内，但很少被普通帧检测到。 剔除
		else if(pMP->GetFoundRatio()<0.25f )
		{
 // 步骤2：将不满足VI-B条件的MapPoint剔除
		// VI-B 条件1：
		// 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%
		// IncreaseFound() / IncreaseVisible(该地图点在视野范围内) < 25%，注意不一定是关键帧。		  
		    pMP->SetBadFlag();
		    lit = mlpRecentAddedMapPoints.erase(lit);//从待查  list中删除
		}
		//
		// 初始三个关键帧 地图点观测次数 不能太少
		// 而且单目的要求更严格，需要三帧都看到
		else if(( (int)nCurrentKFid - (int)pMP->mnFirstKFid) >=2 && pMP->Observations() <= cnThObs)
		{
  // 步骤3：将不满足VI-B条件的MapPoint剔除
            // VI-B 条件2：从该点建立开始，到现在已经过了不小于2帧，
            // 但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格
		    pMP->SetBadFlag();
		    lit = mlpRecentAddedMapPoints.erase(lit);//从待查  list中删除
		}
		else if(((int)nCurrentKFid - (int)pMP->mnFirstKFid ) >= 3)
// 步骤4：从建立该点开始，已经过了3帧(前三帧地图点比较宝贵需要特别检查)，放弃对该MapPoint的检测		  
		    lit = mlpRecentAddedMapPoints.erase(lit);//从待查  list中删除
		else
		    lit++;
	    }
	}


/**
 * @brief 相机运动过程中和共视程度比较高的关键帧通过三角化恢复出一些MapPoints
 *  根据当前关键帧恢复出一些新的地图点，不包括和当前关键帧匹配的局部地图点（已经在ProcessNewKeyFrame中处理）
 *  先处理新关键帧与局部地图点之间的关系，然后对局部地图点进行检查，
 *  最后再通过新关键帧恢复 新的局部地图点：CreateNewMapPoints()
 * 
 * 步骤1：在当前关键帧的 共视关键帧 中找到 共视程度 最高的前nn帧 相邻帧vpNeighKFs
 * 步骤2：遍历和当前关键帧 相邻的 每一个关键帧vpNeighKFs	
 * 步骤3：判断相机运动的基线在（两针间的相机相对坐标）是不是足够长
 * 步骤4：根据两个关键帧的位姿计算它们之间的基本矩阵 F =  inv(K1 转置) * t12 叉乘 R12 * inv(K2)
 * 步骤5：通过帧间词典向量加速匹配，极线约束限制匹配时的搜索范围，进行特征点匹配	
 * 步骤6：对每对匹配点 2d-2d 通过三角化生成3D点,和 Triangulate函数差不多	
 *  步骤6.1：取出匹配特征点
 *  步骤6.2：利用匹配点反投影得到视差角   用来决定使用三角化恢复(视差角较大) 还是 直接2-d点反投影(视差角较小)
 *  步骤6.3：对于双目，利用双目基线 深度 得到视差角
 *  步骤6.4：视差角较大时 使用 三角化恢复3D点
 *  步骤6.4：对于双目 视差角较小时 二维点 利用深度值 反投影 成 三维点    单目的话直接跳过
 *  步骤6.5：检测生成的3D点是否在相机前方
 *  步骤6.6：计算3D点在当前关键帧下的重投影误差  误差较大跳过
 *  步骤6.7：计算3D点在 邻接关键帧 下的重投影误差 误差较大跳过
 *  步骤6.9：三角化生成3D点成功，构造成地图点 MapPoint
 *  步骤6.9：为该MapPoint添加属性 
 *  步骤6.10：将新产生的点放入检测队列 mlpRecentAddedMapPoints  交给 MapPointCulling() 检查生成的点是否合适
 * @see  
 */	
	void LocalMapping::CreateNewMapPoints()
	{
	    // Retrieve neighbor keyframes in covisibility graph
	    int nn = 10;// 双目/深度 共视帧 数量
	    if(mbMonocular)
		nn=20;//单目
		
// 步骤1：在当前关键帧的 共视关键帧 中找到 共视程度 最高的nn帧 相邻帧vpNeighKFs
	    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

	    ORBmatcher matcher(0.6,false);// 描述子匹配器 
            // 当前关键帧 旋转平移矩阵向量
	    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();// 世界---> 当前关键帧
	    cv::Mat Rwc1 = Rcw1.t();// 当前关键帧---> 世界
	    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
	    cv::Mat Tcw1(3,4,CV_32F);// 世界---> 当前关键帧 变换矩阵
	    Rcw1.copyTo(Tcw1.colRange(0,3));
	    tcw1.copyTo(Tcw1.col(3));
	    // 得到当前当前关键帧在世界坐标系中的坐标
	    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
           // 相机内参数
	    const float &fx1 = mpCurrentKeyFrame->fx;
	    const float &fy1 = mpCurrentKeyFrame->fy;
	    const float &cx1 = mpCurrentKeyFrame->cx;
	    const float &cy1 = mpCurrentKeyFrame->cy;
	    const float &invfx1 = mpCurrentKeyFrame->invfx;
	    const float &invfy1 = mpCurrentKeyFrame->invfy;
	    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;//匹配点 查找范围

	    int nnew=0;

	    // Search matches with epipolar restriction and triangulate
// 步骤2：遍历和当前关键帧 相邻的 每一个关键帧vpNeighKFs	    
	    for(size_t i=0; i<vpNeighKFs.size(); i++)
	    {
		if(i>0 && CheckNewKeyFrames())
		    return;

		KeyFrame* pKF2 = vpNeighKFs[i];//关键帧
		// Check first that baseline is not too short
	   // 邻接的关键帧在世界坐标系中的坐标
		cv::Mat Ow2 = pKF2->GetCameraCenter();
	   // 基线向量，两个关键帧间的相机相对坐标
		cv::Mat vBaseline = Ow2-Ow1;
	   // 基线长度	
		const float baseline = cv::norm(vBaseline);
		
// 步骤3：判断相机运动的基线是不是足够长
		if(!mbMonocular)
		{
		    if(baseline < pKF2->mb)
		    continue;// 如果是立体相机，关键帧间距太小时不生成3D点
		}
		else// 单目相机
		{
		   // 邻接关键帧的场景深度中值
		    const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);//中值深度
		    // baseline与景深的比例
		    const float ratioBaselineDepth = baseline/medianDepthKF2; 
                   // 如果特别远(比例特别小)，那么不考虑当前邻接的关键帧，不生成3D点
		    if(ratioBaselineDepth < 0.01)
			continue;
		}

		// Compute Fundamental Matrix
// 步骤4：根据两个关键帧的位姿计算它们之间的基本矩阵	
       // 根据两关键帧的姿态计算两个关键帧之间的基本矩阵 
       // F =  inv(K1 转置)*E*inv(K2) = inv(K1 转置) * t12 叉乘 R12 * inv(K2)
		cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

		// Search matches that fullfil epipolar constraint
// 步骤5：通过帧间词典向量加速匹配，极线约束限制匹配时的搜索范围，进行特征点匹配		
		vector<pair<size_t,size_t> > vMatchedIndices;// 特征匹配候选点
		matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

         	 // 相邻关键帧 旋转平移矩阵向量
		cv::Mat Rcw2 = pKF2->GetRotation();
		cv::Mat Rwc2 = Rcw2.t();
		cv::Mat tcw2 = pKF2->GetTranslation();
		cv::Mat Tcw2(3,4,CV_32F);// 转换矩阵
		Rcw2.copyTo(Tcw2.colRange(0,3));
		tcw2.copyTo(Tcw2.col(3));
		// 相机内参
		const float &fx2 = pKF2->fx;
		const float &fy2 = pKF2->fy;
		const float &cx2 = pKF2->cx;
		const float &cy2 = pKF2->cy;
		const float &invfx2 = pKF2->invfx;
		const float &invfy2 = pKF2->invfy;

		// Triangulate each match
		// 三角化每一个匹配点对
// 步骤6：对每对匹配点 2d-2d 通过三角化生成3D点,和 Triangulate函数差不多		
		const int nmatches = vMatchedIndices.size();
		for(int ikp=0; ikp<nmatches; ikp++)
		{
	 // 步骤6.1：取出匹配特征点	  
		    const int &idx1 = vMatchedIndices[ikp].first; // 当前匹配对在当前关键帧中的索引
		    const int &idx2 = vMatchedIndices[ikp].second;// 当前匹配对在邻接关键帧中的索引
		    //当前关键帧 特征点 和 右图像匹配点横坐标
		    const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
		    const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
		    bool bStereo1 = kp1_ur >= 0;//右图像匹配点横坐标>=0是双目/深度相机
		     // 邻接关键帧 特征点 和 右图像匹配点横坐标
		    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
		    const float kp2_ur = pKF2->mvuRight[idx2];
		    bool bStereo2 = kp2_ur >= 0;

	 // 步骤6.2：利用匹配点反投影得到视差角    
		    // Check parallax between rays 
		    // 相机归一化平面上的点坐标
		    cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x - cx1)*invfx1, (kp1.pt.y - cy1)*invfy1, 1.0);
		    cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x - cx2)*invfx2, (kp2.pt.y - cy2)*invfy2, 1.0);
		    
		    // 由相机坐标系转到世界坐标系，得到视差角余弦值
		    cv::Mat ray1 = Rwc1*xn1;// 相机坐标系 ------> 世界坐标系
		    cv::Mat ray2 = Rwc2*xn2;
		    // 向量a × 向量b / （向量a模 × 向量吧模） = 夹角余弦值
		    const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));
		    
		    // 加1是为了让cosParallaxStereo随便初始化为一个很大的值
		    float cosParallaxStereo = cosParallaxRays+1;
		    float cosParallaxStereo1 = cosParallaxStereo;
		    float cosParallaxStereo2 = cosParallaxStereo;
	 // 步骤6.3：对于双目，利用双目基线 深度 得到视差角
		    if(bStereo1)
			cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
		    else if(bStereo2)
			cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));
		    // 得到双目观测的视差角
		    cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
		    
	 // 步骤6.4：三角化恢复3D点
		    cv::Mat x3D;
		  // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表明视差角正常
                  // cosParallaxRays < cosParallaxStereo表明视差角很小
                  // 视差角度小时用三角法恢复3D点，视差角大时用双目恢复3D点（双目以及深度有效）
		    if(cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
		    {
			// Linear Triangulation Method
		      // p1 = k × [R1 t1] × D       k逆 × p1 =  [R1 t1] × D     x1 = T1 × D    x1叉乘x1 = x1叉乘T1 × D = 0
		      // p2 = k × [ R2 t2]  × D     k逆 × p2 =  [R2 t2] × D     x2 = T2 × D    x2叉乘x2 = x2叉乘T2 × D = 0
		      //
		      //p = ( x,y,1)
		      //其叉乘矩阵为
		      //  叉乘矩阵 = [0  -1  y;                T0
		      //                       1   0  -x;           *  T1  *D  ===>( y * T2 - T1 ) *D = 0
		      //                      -y   x  0 ] 		    T2                  ( x * T2 - T0 ) *D = 0
		      //一个点两个方程  两个点 四个方程  A × D =0  求三维点 D  对 A奇异值分解
			cv::Mat A(4,4,CV_32F);
			A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
			A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
			A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
			A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);

			cv::Mat w,u,vt;
			cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

			x3D = vt.row(3).t();

			if(x3D.at<float>(3)==0)
			    continue;

			// Euclidean coordinates
			x3D = x3D.rowRange(0,3) / x3D.at<float>(3);//其次点坐标 除去尺度

		    }
	   //   步骤6.4：对于双目 视差角较小时 二维点 利用深度值 反投影 成 三维点
		    else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)// 双目 视差角 小
		    {
			x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);// 二维点 反投影 成 三维点               
		    }
		    else if(bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
		    {
			x3D = pKF2->UnprojectStereo(idx2);
		    }
	       // 单目 视差角 较小时 生成不了三维点
		    else
			continue; //没有双目/深度 且两针视角差太小  三角测量也不合适 得不到三维点 No stereo and very low parallax

		    cv::Mat x3Dt = x3D.t();
		    
	 // 步骤6.5：检测生成的3D点是否在相机前方
		    //Check triangulation in front of cameras
		    float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);// 只算z坐标值
		    if(z1<= 0)
			continue;
		    float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
		    if(z2 <= 0)
			continue;
		    
         // 步骤6.6：计算3D点在当前关键帧下的重投影误差
		    //Check reprojection error in first keyframe
		    const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];//误差 分布参数
		    const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);//相机归一化坐标
		    const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
		    const float invz1 = 1.0/z1;
		    if(!bStereo1)
		    {// 单目
			float u1 = fx1*x1*invz1 + cx1;//像素坐标
			float v1 = fy1*y1*invz1 + cy1;
			float errX1 = u1 - kp1.pt.x;
			float errY1 = v1 - kp1.pt.y;
			if((errX1*errX1+errY1*errY1) > 5.991*sigmaSquare1)
			    continue;//投影误差过大 跳过
		    }
		    else
		    {// 双目 / 深度 相机   有右图像匹配点横坐标差值
			float u1 = fx1*x1*invz1+cx1;
			float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;//左图像坐标值 - 视差 =   右图像匹配点横坐标 
			float v1 = fy1*y1*invz1+cy1;
			float errX1 = u1 - kp1.pt.x;
			float errY1 = v1 - kp1.pt.y;
			float errX1_r = u1_r - kp1_ur;
			// 基于卡方检验计算出的阈值（假设测量有一个一个像素的偏差）
			if((errX1*errX1 + errY1*errY1 + errX1_r*errX1_r) > 7.8*sigmaSquare1)
			    continue;//投影误差过大 跳过
		    }
         // 步骤6.7：计算3D点在 邻接关键帧 下的重投影误差
		    //Check reprojection error in second keyframe
		    const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
		    const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
		    const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
		    const float invz2 = 1.0/z2;
		    if(!bStereo2)
		    {// 单目
			float u2 = fx2*x2*invz2+cx2;
			float v2 = fy2*y2*invz2+cy2;
			float errX2 = u2 - kp2.pt.x;
			float errY2 = v2 - kp2.pt.y;
			if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
			    continue;//投影误差过大 跳过
		    }
		    else
		    {// 双目 / 深度 相机   有右图像匹配点横坐标差值
			float u2 = fx2*x2*invz2+cx2;
			float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;//左图像坐标值 - 视差 =   右图像匹配点横坐标 
			float v2 = fy2*y2*invz2+cy2;
			float errX2 = u2 - kp2.pt.x;
			float errY2 = v2 - kp2.pt.y;
			float errX2_r = u2_r - kp2_ur;
			if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
			    continue;//投影误差过大 跳过
		    }
		    
           // 步骤6.8：检查尺度连续性
		    //Check scale consistency
		    cv::Mat normal1 = x3D-Ow1;//  世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
		    float dist1 = cv::norm(normal1);// 模长
		    cv::Mat normal2 = x3D-Ow2;
		    float dist2 = cv::norm(normal2);
		    if(dist1==0 || dist2==0)
			continue;// 模长为0 跳过
                   // ratioDist是不考虑金字塔尺度下的距离比例
		    const float ratioDist = dist2/dist1;
		   // 金字塔尺度因子的比例
		    const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];
		    /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
			continue;*/
		    // 深度比值和 两幅图像下的金字塔层级比值应该相差不大
		    // |ratioDist/ratioOctave |<ratioFactor
		    if(ratioDist * ratioFactor<ratioOctave || ratioDist > ratioOctave*ratioFactor)
			continue;
		    
	  // 步骤6.9：三角化生成3D点成功，构造成地图点 MapPoint
		    // Triangulation is succesfull
		    MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
		    

            // 步骤6.9：为该MapPoint添加属性：
	       // a.观测到该MapPoint的关键帧 
		    pMP->AddObservation(mpCurrentKeyFrame,idx1); // 地图点 添加观测帧  
		    pMP->AddObservation(pKF2,idx2);// 
		    mpCurrentKeyFrame->AddMapPoint(pMP,idx1);// 关键帧 添加地图点
		    pKF2->AddMapPoint(pMP,idx2);
              // b.该MapPoint的描述子
		    pMP->ComputeDistinctiveDescriptors();
               // c.该MapPoint的平均观测方向和深度范围
		    pMP->UpdateNormalAndDepth();
               // d.地图添加地图点
		    mpMap->AddMapPoint(pMP);
	
	
            // 步骤6.10：将新产生的点放入检测队列 mlpRecentAddedMapPoints
                  // 这些MapPoints都会经过MapPointCulling函数的检验
		    mlpRecentAddedMapPoints.push_back(pMP);

		    nnew++;
		}
	    }
	}
	

/**
 * @brief     检查并融合 当前关键帧 与 相邻帧（一级二级相邻帧）重复的地图点 MapPoints
 * 步骤1：获得当前关键帧在covisibility帧连接图中权重排名前nn的一级邻接关键帧(按观测到当前帧地图点次数选取)
 * 步骤2：获得当前关键帧在 其一级相邻帧 在 covisibility图中权重排名前5 的二级邻接关键帧
 * 步骤3：将当前帧的 地图点MapPoints 分别与 其一级二级相邻帧的 地图点 MapPoints 进行融合(保留观测次数最高的)
 * 步骤4：找到一级二级相邻帧所有的地图点MapPoints 与当前帧 的  地图点MapPoints 进行融合
 * 步骤5：更新当前帧 地图点 MapPoints 的描述子，深度，观测主方向等属性
 * 步骤5：更新当前 与其它帧的连接关系 (  观测到互相的地图点的次数等信息 )
 * @return  无
 */	
	void LocalMapping::SearchInNeighbors()
	{
	    // Retrieve neighbor keyframes
 // 步骤1：获得当前关键帧在covisibility图中权重排名前nn的一级邻接关键帧
          // 找到当前帧一级相邻与二级相邻关键帧
	    int nn = 10;
	    if(mbMonocular)
		nn=20;//单目 多找一些	
	   // 一级相邻	
	    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
	    vector<KeyFrame*> vpTargetKFs;// 最后合格的一级二级相邻关键帧
	    // 遍历每一个 一级相邻帧
	    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
	    {
		KeyFrame* pKFi = *vit;// 一级相邻关键帧
		if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)//坏帧  或者 已经加入过
		    continue;// 跳过
		vpTargetKFs.push_back(pKFi);// 加入 最后合格的相邻关键帧
		pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;// 已经做过相邻匹配  标记已经加入
		
 // 步骤2：获得当前关键帧在 其一级相邻帧的  covisibility图中权重排名前5的二级邻接关键帧
	        // 二级相邻	
		// Extend to some second neighbors
		const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
		// 遍历每一个 二级相邻帧
		for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
		{
		    KeyFrame* pKFi2 = *vit2;// 二级相邻关键帧
		    if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
			continue;// 二级相邻关键帧是坏帧 在一级时已经加入 或者 又找回来了找到当前帧了 跳过
		    vpTargetKFs.push_back(pKFi2);
		}
	    }

// 步骤3：将当前帧的 地图点MapPoints 分别与 其一级二级相邻帧的 地图点 MapPoints 进行融合
	    // Search matches by projection from current KF in target KFs
	    ORBmatcher matcher;
	    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();//与当前帧 匹配的地图点
	    // vector<KeyFrame*>::iterator
	    for(auto  vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
	    {
		KeyFrame* pKFi = *vit;//其一级二级相邻帧
		// 投影当前帧的MapPoints到相邻关键帧pKFi中，在附加区域搜索匹配关键点，并判断是否有重复的MapPoints
		// 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
		// 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint		
		matcher.Fuse(pKFi,vpMapPointMatches);
	    }
	    
	    
// 步骤4：将一级二级相邻帧所有的地图点MapPoints 与当前帧（的MapPoints）进行融合
            // 遍历每一个一级邻接和二级邻接关键帧 找到所有的地图点
	    // Search matches by projection from target KFs in current KF
	    // 用于存储一级邻接和二级邻接关键帧所有MapPoints的集合
	    vector<MapPoint*> vpFuseCandidates;// 一级二级相邻帧所有地图点
	    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());// 帧数量 × 每一帧地图点数量
            // vector<KeyFrame*>::iterator
	    for(auto vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
	    {
		KeyFrame* pKFi = *vitKF;//其一级二级相邻帧
		vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();//地图点
		
		// vector<MapPoint*>::iterator
		for(auto vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
		{
		    MapPoint* pMP = *vitMP;//  一级二级相邻帧 的每一个地图点
		    if(!pMP)
			continue;
		    if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
			continue;
		    // 加入集合，并标记 已经加入
		    pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId; //标记 已经加
		    vpFuseCandidates.push_back(pMP); // 加入 一级二级相邻帧 地图点 集合
		}
	    }
	    
            //一级二级相邻帧 所有的 地图点 与当前帧 融合 
            		// 投影 地图点MapPoints到当前帧上，在附加区域搜索匹配关键点，并判断是否有重复的地图点
		        // 1.如果MapPoint能匹配当前帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
		        // 2.如果MapPoint能匹配当前帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
	    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);

// 步骤5：更新当前帧MapPoints的描述子，深度，观测主方向等属性
	    // Update points
	    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();//当前帧 所有的 匹配地图点
	    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
	    {
		MapPoint* pMP=vpMapPointMatches[i];//当前帧 每个关键点匹配的地图点
		if(pMP)//存在
		{
		    if(!pMP->isBad())//非 坏点
		    {
			pMP->ComputeDistinctiveDescriptors();// 更新 地图点的描述子(在所有观测在的描述子中选出最好的描述子)
			pMP->UpdateNormalAndDepth();          // 更新平均观测方向和观测距离
		    }
		}
	    }
// 步骤5：更新当前帧的MapPoints后 更新与其它帧的连接关系 观测到互相的地图点的次数等信息
            // 更新covisibility图
	    // Update connections in covisibility graph
	    mpCurrentKeyFrame->UpdateConnections();
	}
	

/**
 * @brief    关键帧剔除
 *  在Covisibility Graph 关键帧连接图 中的关键帧，
 *  其90%以上的地图点MapPoints能被其他关键帧（至少3个）观测到，
 *  则认为该关键帧为冗余关键帧。
 * @param  pKF1 关键帧1
 * @param  pKF2 关键帧2
 * @return 两个关键帧之间的基本矩阵 F
 */	
void LocalMapping::KeyFrameCulling()
	{
	    // Check redundant keyframes (only local keyframes)
	    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
	    // in at least other 3 keyframes (in the same or finer scale)
	    // We only consider close stereo points
	  
// 步骤1：根据Covisibility Graph 关键帧连接 图提取当前帧的 所有共视关键帧(关联帧)	  
	    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
	    
            // vector<KeyFrame*>::iterator
           // 对所有的局部关键帧进行遍历	    
	    for(auto  vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
	    {
		KeyFrame* pKF = *vit;// 当前帧的每一个 局部关联帧
		if(pKF->mnId == 0)//第一帧关键帧为 初始化世界关键帧 跳过
		    continue;
		
// 步骤2：提取每个共视关键帧的 地图点 MapPoints		
		const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();// 局部关联帧 匹配的 地图点

		int nObs = 3;
		const int thObs=nObs; //3
		int nRedundantObservations=0;
		int nMPs=0;
		
// 步骤3：遍历该局部关键帧的MapPoints，判断是否90%以上的MapPoints能被其它关键帧（至少3个）观测到		
		for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
		{
		    MapPoint* pMP = vpMapPoints[i];// 该局部关键帧的 地图点 MapPoints
		    if(pMP)
		    {
			if(!pMP->isBad())
			{
			    if(!mbMonocular)// 双目/深度
			    {  // 对于双目，仅考虑近处的MapPoints，不超过mbf * 35 / fx
				if(pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
				    continue;
			    }

			    nMPs++;
			    // 地图点 MapPoints 至少被三个关键帧观测到
			    if(pMP->Observations() > thObs)// 观测帧个数 > 3
			    {
				const int &scaleLevel = pKF->mvKeysUn[i].octave;// 金字塔层数
				const map<KeyFrame*, size_t> observations = pMP->GetObservations();// 局部 观测关键帧地图
				int nObs=0;
				for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
				{
				    KeyFrame* pKFi = mit->first;
				    if(pKFi==pKF)// 跳过 原地图点的帧
					continue;
				    const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;// 金字塔层数
				    
                                   // 尺度约束，要求MapPoint在该局部关键帧的特征尺度大于（或近似于）其它关键帧的特征尺度
				    if(scaleLeveli <= scaleLevel+1)
				    {
					nObs++;
					if(nObs >= thObs)
					    break;
				    }
				}
				if(nObs >= thObs)
				{// 该MapPoint至少被三个关键帧观测到
				    nRedundantObservations++;
				}
			    }
			}
		    }
		}  
		
 // 步骤4：该局部关键帧90%以上的MapPoints能被其它关键帧（至少3个）观测到，则认为是冗余关键帧
		if(nRedundantObservations > 0.9*nMPs)
		    pKF->SetBadFlag();
	    }
	}
	
/**
 * @brief    根据两关键帧的姿态计算两个关键帧之间的基本矩阵 
 *                 F =  inv(K1 转置)*E*inv(K2) = inv(K1 转置)*t叉乘R*inv(K2)
 * @param  pKF1 关键帧1
 * @param  pKF2 关键帧2
 * @return 两个关键帧之间的基本矩阵 F
 */
	cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
	{
    // Essential Matrix: t12叉乘R12
    // Fundamental Matrix: inv(K1转置)*E*inv(K2)	  
	    cv::Mat R1w = pKF1->GetRotation();   // Rc1w
	    cv::Mat t1w = pKF1->GetTranslation();
	    cv::Mat R2w = pKF2->GetRotation();   // Rc2w
	    cv::Mat t2w = pKF2->GetTranslation();// t c2 w

	    cv::Mat R12 = R1w*R2w.t();// R12 =Rc1w *  Rwc2 // c2 -->w --->c1
	    cv::Mat t12 = -R12*t2w + t1w; // tw2  + t1w    // c2 -->w --->c1

	    // t12 的叉乘矩阵
	    cv::Mat t12x = SkewSymmetricMatrix(t12);

	    const cv::Mat &K1 = pKF1->mK;
	    const cv::Mat &K2 = pKF2->mK;

	    return K1.t().inv()*t12x*R12*K2.inv();
	}

/**
 * @brief    请求停止局部建图线程  设置停止标志
 * @return 无
 */	
	void LocalMapping::RequestStop()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    mbStopRequested = true;//局部建图 请求停止
	    unique_lock<mutex> lock2(mMutexNewKFs);
	    mbAbortBA = true;//停止BA 优化
	}
	
/**
 * @brief    停止局部建图线程  设置停止标志
 * @return 无
 */	
	bool LocalMapping::Stop()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    if(mbStopRequested && !mbNotStop)
	    {
		mbStopped = true;
		cout << "局部建图停止 Local Mapping STOP" << endl;
		return true;
	    }
	    return false;
	}
	
/**
 * @brief    检查局部建图线程 是否停止
 * @return 是否停止标志
 */	
	bool LocalMapping::isStopped()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    return mbStopped;
	}
	
/**
 * @brief   返回 请求停止局部建图线程   标志
 * @return 返回 请求停止局部建图线程   标志
 */	
	bool LocalMapping::stopRequested()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    return mbStopRequested;
	}
	
/**
 * @brief    释放局部建图线程  
 * @return  无
 */	
	void LocalMapping::Release()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    unique_lock<mutex> lock2(mMutexFinish);
	    if(mbFinished)
		return;
	    mbStopped = false;
	    mbStopRequested = false;
	    // list<KeyFrame*>::iterator
	    for(auto lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
		delete *lit;//删除关键帧
	    mlNewKeyFrames.clear();

	    cout << "局部建图释放 Local Mapping RELEASE" << endl;
	}

/**
 * @brief    返回 可以接收新的一个关键帧标志
 * @return 是否 可以接收新的一个关键帧
 */		
	bool LocalMapping::AcceptKeyFrames()
	{
	    unique_lock<mutex> lock(mMutexAccept);
	    return mbAcceptKeyFrames;
	}
	
/**
 * @brief    设置 可以接收新的一个关键帧标志
 * @return 无
 */
	void LocalMapping::SetAcceptKeyFrames(bool flag)
	{
	    unique_lock<mutex> lock(mMutexAccept);
	    mbAcceptKeyFrames=flag;
	}
	
/**
 * @brief    设置不要停止标志 
 * @return  成功与否 
 */
	bool LocalMapping::SetNotStop(bool flag)
	{
	    unique_lock<mutex> lock(mMutexStop);

	    if(flag && mbStopped)//  在已经停止的情况下 设置不要停止   错误
		return false;

	    mbNotStop = flag;

	    return true;
	}
/**
 * @brief    停止 全局优化 BA
 * @return 是否 可以接收新的一个关键帧
 */
	void LocalMapping::InterruptBA()
	{
	    mbAbortBA = true;
	}

	
	
/**
 * @brief   计算向量的 叉乘矩阵     变叉乘为 矩阵乘
 * @return 该向量的叉乘矩阵
 */	
	cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
	{
// 向量 t=(a1 a2 a3)	t叉乘A
// 等于 向量t的叉乘矩阵 * A
//  t的叉乘矩阵
//|0     -a3  a2 |
//|a3     0   -a1|
//|-a2   a1    0 |
	    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
		    v.at<float>(2),               0,-v.at<float>(0),
		    -v.at<float>(1),  v.at<float>(0),              0);
	}
	
/**
 * @brief    请求重置
 * @return  无
 */
	void LocalMapping::RequestReset()
	{
	    {
		unique_lock<mutex> lock(mMutexReset);
		mbResetRequested = true;
	    }

	    while(1)
	    {
		{
		    unique_lock<mutex> lock2(mMutexReset);
		    if(!mbResetRequested)
			break;
		}
		usleep(3000);
	    }
	}
	
/**
 * @brief    重置线程
 * @return  无
 */
	void LocalMapping::ResetIfRequested()
	{
	    unique_lock<mutex> lock(mMutexReset);
	    if(mbResetRequested)
	    {
		mlNewKeyFrames.clear();
		mlpRecentAddedMapPoints.clear();
		mbResetRequested=false;
	    }
	}
	
	void LocalMapping::RequestFinish()
	{
	    unique_lock<mutex> lock(mMutexFinish);
	    mbFinishRequested = true;
	}

	bool LocalMapping::CheckFinish()
	{
	    unique_lock<mutex> lock(mMutexFinish);
	    return mbFinishRequested;
	}

	void LocalMapping::SetFinish()
	{
	    unique_lock<mutex> lock(mMutexFinish);
	    mbFinished = true;    
	    unique_lock<mutex> lock2(mMutexStop);
	    mbStopped = true;
	}

	bool LocalMapping::isFinished()
	{
	    unique_lock<mutex> lock(mMutexFinish);
	    return mbFinished;
	}

} //namespace ORB_SLAM
