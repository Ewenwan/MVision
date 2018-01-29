/**
* This file is part of ORB-SLAM2.
* 地图点  普通帧上的地图点
*               关键帧上的地图点
* 创建 地图点    观测帧集合   最优的描述子
* 环点检测 被观测次数 小于2  删除地图点
* 地图点距离 参考帧相机中心的 相对 坐标
* 地图点 相对参考帧 相机中心  在 图像金字塔上 每层 各个尺度空间的距离
*/

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

	long unsigned int MapPoint::nNextId=0;
	mutex MapPoint::mGlobalMutex;// 全局 线程锁

	// 创建关键帧地图点   世界坐标点     所属关键帧    所属地图
	MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
	    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
	    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
	    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
	    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
	{
	    Pos.copyTo(mWorldPos);// 世界坐标点 拷贝到类内
	    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

	    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
	    mnId=nNextId++;
	}

	// 创建普通帧地图点    世界坐标点       所属地图   所属普通 帧      帧id
	MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
	    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
	    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
	    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
	    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
	{
	    Pos.copyTo(mWorldPos);// 世界坐标点 拷贝到类内
	    cv::Mat Ow = pFrame->GetCameraCenter();// 普通帧 相机 光行坐标（单目 左 相机）
	    mNormalVector = mWorldPos - Ow;// 点 相对于 相机光心坐标 
	    mNormalVector = mNormalVector/cv::norm(mNormalVector);

	    cv::Mat PC = Pos - Ow;// 相对于 相机光心坐标
	    const float dist = cv::norm(PC);// 点坐标 距离 相机光心的距离 
	    const int level = pFrame->mvKeysUn[idxF].octave;// 关键点所在 金字塔的 层级数
	    const float levelScaleFactor =  pFrame->mvScaleFactors[level];// // 关键点所在 金字塔的 尺度因子
	    const int nLevels = pFrame->mnScaleLevels;

	    mfMaxDistance = dist*levelScaleFactor;// 最大距离
	    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];// 最小距离

	    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);// 描述子

	    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
	    mnId=nNextId++;
	}

	void MapPoint::SetWorldPos(const cv::Mat &Pos)
	{
	    unique_lock<mutex> lock2(mGlobalMutex);
	    unique_lock<mutex> lock(mMutexPos);
	    Pos.copyTo(mWorldPos);
	}

	cv::Mat MapPoint::GetWorldPos()
	{
	    unique_lock<mutex> lock(mMutexPos);
	    return mWorldPos.clone();// 复制
	}

	cv::Mat MapPoint::GetNormal()
	{
	    unique_lock<mutex> lock(mMutexPos);
	    return mNormalVector.clone();// 点 相对于 相机光心坐标 
	}

	KeyFrame* MapPoint::GetReferenceKeyFrame()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return mpRefKF;// 地图点的 参考帧
	}

	// 添加能够观测到 的关键帧  对应的 关键点id
	// 更新能够 观测到改点  的帧数量
	void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    if(mObservations.count(pKF))// pKF已经出现了
		return;
	    mObservations[pKF]=idx;// 还没有   则添加 观测关键帧
	    
	  // 更新能够 观测到改点  的帧数量
	  if(pKF->mvuRight[idx] >= 0)// 有 匹配点
		nObs += 2;// 还有匹配点 匹配点那一帧 也观测到了 该点
	    else
		nObs++; // 本帧 自身 观测到
	}

	// 删除该点 的观测观测帧
	// 更新该点被观测到 的次数 -2 (有匹配点对)/ -1(无匹配点对)
	// 更新参考帧
	// 坏点检测  删除 坏点所有的观测帧指针 观测帧 内 的 与该地图点相关的匹配点对
	void MapPoint::EraseObservation(KeyFrame* pKF)
	{
	    bool bBad=false;
	    {
		unique_lock<mutex> lock(mMutexFeatures);
		if(mObservations.count(pKF))
		{
		    int idx = mObservations[pKF];// 观测帧 id
		    if(pKF->mvuRight[idx] >= 0)
			nObs-=2;// 删除该关键帧时 其 匹配帧 也没了  观测次数 -2
		    else
			nObs--;// 没有匹配帧   减去1

		    mObservations.erase(pKF);// 删除 观测帧

		    if(mpRefKF==pKF)// 删除的观测帧为 参考帧   参考值设为 观测第一帧
			mpRefKF=mObservations.begin()->first;

		    // If only 2 observations or less, discard point
		    if(nObs<=2)// 改点被观测的次数 过少
			bBad=true;// 不好的点 坏点
		}
	    }
	    if(bBad)//删除 坏点
		SetBadFlag();//删除 该地图点 的 观测帧 指针  、观测帧内于此点相关的 匹配点对 、该点也删除
	}
	// 得到该地图点的 观测帧
	map<KeyFrame*, size_t> MapPoint::GetObservations()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return mObservations;
	}
	// 返回改点被观测到 的次数
	int MapPoint::Observations()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return nObs;
	}

	// 地图点被观测到的次数过少 该点不具有参考性 坏点
	// 删除 该地图点 的 观测帧 指针 其他点 也可能被这些观测点观测到  对象不能删除
	// 删除在 观测帧 内 的 与改地图点相关的匹配点对
	void MapPoint::SetBadFlag()
	{
	    map<KeyFrame*,size_t> obs;
	    {
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		mbBad=true;
		obs = mObservations;// 保存改点对应的 观测帧
		mObservations.clear();// 清除观测帧
	    }
	    // map<KeyFrame*,size_t>::iterator mit 
	    for(auto mit = obs.begin(), mend=obs.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKF = mit->first;// 关键帧 指针
		pKF->EraseMapPointMatch(mit->second);// 在对应关键帧 内删除对应改点的匹配点
	    }

	    mpMap->EraseMapPoint(this);// 删除这个 点
	}

	// 代替点
	MapPoint* MapPoint::GetReplaced()
	{
	    unique_lock<mutex> lock1(mMutexFeatures);
	    unique_lock<mutex> lock2(mMutexPos);
	    return mpReplaced;
	}
	// 地图点的 替换点   输入可替换本地图点的 另一个地图点
	void MapPoint::Replace(MapPoint* pMP)
	{
	    if(pMP->mnId == this->mnId)//同一个点 直接返回
		return;

	    int nvisible, nfound;
	    map<KeyFrame*,size_t> obs;
	    {
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		obs=mObservations;// 本点 所有的 观测帧
		mObservations.clear();// 清空 观测帧 指针
		mbBad=true;//坏点
		nvisible = mnVisible;// 可见次数？
		nfound = mnFound;// 跟踪到 次数？
		mpReplaced = pMP;
	    }
	// map<KeyFrame*,size_t>::iterator mit
	    for( auto mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)//对于 原来点的所有观测帧 检测与替代点关系 
	    {
		// Replace measurement in keyframe
		KeyFrame* pKF = mit->first;// 观测帧 第一帧

		if(!pMP->IsInKeyFrame(pKF))// 没有在 原来点 的观测 关键帧 内
		{
		    pKF->ReplaceMapPointMatch(mit->second, pMP);
		    pMP->AddObservation(pKF,mit->second);
		}
		else
		{
		    pKF->EraseMapPointMatch(mit->second);
		}
	    }
	    pMP->IncreaseFound(nfound);
	    pMP->IncreaseVisible(nvisible);
	    pMP->ComputeDistinctiveDescriptors();

	    mpMap->EraseMapPoint(this);// 删除原来的点
	}

	// 是否是坏点  被观测到的次数小鱼2
	bool MapPoint::isBad()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    unique_lock<mutex> lock2(mMutexPos);
	    return mbBad;
	}

	// 可观测到次数 +n
	void MapPoint::IncreaseVisible(int n)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    mnVisible+=n;
	}

	// 可跟踪到次数 +n
	void MapPoint::IncreaseFound(int n)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    mnFound+=n;
	}

	// 可跟踪到次数/可见次数
	float MapPoint::GetFoundRatio()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return static_cast<float>(mnFound)/mnVisible;
	}

	// 得到最具代表性的 描述子
	// 每个地图点（世界中的一个点）在各个观测帧上 都有一个描述子
	// 各个描述子 之间的 orb 二进制字符串汉明匹配距离
	// 计算每个描述子和其他描述子 距离的中值
	// 最小的距离中值  对于的描述子
	void MapPoint::ComputeDistinctiveDescriptors()
	{
	// 所有观测帧------------------------------
	    map<KeyFrame*,size_t> observations;
	    {
		unique_lock<mutex> lock1(mMutexFeatures);
		if(mbBad)
		    return;
		observations=mObservations;// 所有观测帧
	    }
	    if(observations.empty())
		return;
	// 所有 描述子----------------------
	      // Retrieve all observed descriptors
	    vector<cv::Mat> vDescriptors;//该地图点 在 所有 观测帧 上的 描述子 集合   
	    vDescriptors.reserve(observations.size());   
	    // map<KeyFrame*,size_t>::iterator mit
	    for( auto mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKF = mit->first;

		if(!pKF->isBad())
		    vDescriptors.push_back(pKF->mDescriptors.row(mit->second));// 该点在 所有观测帧 下的 描述子
	    }
	    if(vDescriptors.empty())
		return;
	// 该点 所有描述子 之间 的距离------------------------
	    // Compute distances between them
	    const size_t N = vDescriptors.size();// 该点 中的 描述子 个数
	    float Distances[N][N];// 该点 N个描述子之间的 距离
	    for(size_t i=0;i<N;i++)
	    {
		Distances[i][i]=0;
		for(size_t j=i+1;j<N;j++)
		{
		    int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);// 汉明匹配距离
		    Distances[i][j]=distij;
		    Distances[j][i]=distij;
		}
	    }
	// 最小中值距离    
	  // 计算每个描述子和 其他描述子之间的距离  排序  求中值 距离
	  // 所有描述子  最小的中值距离 对应的  观测帧 上 对应 的描述子
	  // Take the descriptor with least median distance to the rest   
	    int BestMedian = INT_MAX;
	    int BestIdx = 0;
	    for(size_t i=0;i<N;i++)
	    {
		vector<int> vDists(Distances[i],Distances[i]+N);//每个描述子 和其他描述子之间的距离
		sort(vDists.begin(),vDists.end());// 排序
		int median = vDists[0.5*(N-1)];// 中值距离
		if(median<BestMedian)
		{
		    BestMedian = median;//保留最小的 中值距离
		    BestIdx = i;
		}
	    }
	// 最小中值距离  对于对应的 描述子
	    {
		unique_lock<mutex> lock(mMutexFeatures);
		mDescriptor = vDescriptors[BestIdx].clone();// 和其他描述子最想的 描述子
	    }
	}

	// 得到地图点  在 所有观测帧中的 最具代表性的 描述子
	cv::Mat MapPoint::GetDescriptor()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return mDescriptor.clone();
	}

	// 返回 给定帧 在 该地图点 的 观测帧集合中的 位置（）
	int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    if(mObservations.count(pKF))// 找该关键帧 在 观测关键帧集合 中的位置
		return mObservations[pKF];
	    else
		return -1;
	}

	// 给定关键帧 是否在 该点的 观测帧集合内
	bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return (mObservations.count(pKF));
	}

	// 更新 地图点 相对 观测帧 相机中心 单位化坐标
	// 更新 地图点 在参考帧下 各个金字塔层级 下的  最小最大距离
	void MapPoint::UpdateNormalAndDepth()
	{
	    map<KeyFrame*,size_t> observations;// 观测帧
	    KeyFrame* pRefKF;//参考关键帧
	    cv::Mat Pos;//世界 3D坐标
	    {
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		if(mbBad)
		    return;
		observations=mObservations;// 观测帧
		pRefKF=mpRefKF;//参考关键帧
		Pos = mWorldPos.clone();//世界 3D坐标
	    }
	    if(observations.empty())
		return;

	// 计算 3D世界 地图点 在 各个观测帧 相机下 的 相对相机中的 的 单位化 相对坐标----------------
	    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
	    int n=0;
	    // map<KeyFrame*,size_t>::iterator mit
	    for(auto mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKF = mit->first;// 观测 关键帧
		cv::Mat Owi = pKF->GetCameraCenter();// 相机坐标中心
		cv::Mat normali = mWorldPos - Owi;//3D 点里该观测帧 相机中心的 相对坐标
		normal = normal + normali/cv::norm(normali);//单位化相对坐标
		n++;
	    }
	// 在参考帧下 各个图像金字塔下 的距离----------------------
	    cv::Mat PC = Pos - pRefKF->GetCameraCenter();// 相对于参考帧 相机中心 的 相对坐标
	    const float dist = cv::norm(PC);// 3D点相对于 参考帧相机中心的 距离
	    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;// 在 参考帧下 图像金字塔 中的层级位置
	    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];// 对应层级下 的尺度因子
	    const int nLevels = pRefKF->mnScaleLevels;
	    {
		unique_lock<mutex> lock3(mMutexPos);
		mfMaxDistance = dist*levelScaleFactor;// 原来的距离 在 对于层级尺度下的 距离
		mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];// 
		mNormalVector = normal/n;
	    }
	}

	float MapPoint::GetMinDistanceInvariance()
	{
	    unique_lock<mutex> lock(mMutexPos);
	    return 0.8f*mfMinDistance;// 各个图像金字塔下 的距离 最小距离
	}

	float MapPoint::GetMaxDistanceInvariance()
	{
	    unique_lock<mutex> lock(mMutexPos);
	    return 1.2f*mfMaxDistance;// 各个图像金字塔下 的距离 最大距离
	}

	int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
	{
	    float ratio;
	    {
		unique_lock<mutex> lock(mMutexPos);
		ratio = mfMaxDistance/currentDist;
	    }

	    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
	    if(nScale<0)
		nScale = 0;
	    else if(nScale>=pKF->mnScaleLevels)
		nScale = pKF->mnScaleLevels-1;

	    return nScale;// 预测尺度
	}

	int MapPoint::PredictScale(const float &currentDist, Frame* pF)
	{
	    float ratio;
	    {
		unique_lock<mutex> lock(mMutexPos);
		ratio = mfMaxDistance/currentDist;
	    }

	    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
	    if(nScale<0)
		nScale = 0;
	    else if(nScale>=pF->mnScaleLevels)
		nScale = pF->mnScaleLevels-1;

	    return nScale;
	}



} //namespace ORB_SLAM
