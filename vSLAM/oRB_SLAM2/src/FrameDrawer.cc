/**
* This file is part of ORB-SLAM2.
* 获取帧 显示 图像+关键点====
*/

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));// 初始化一个空的三通道图像
}

cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // 初始化参考帧关键点 Initialization: KeyPoints in reference frame
    vector<int> vMatches;          // 匹配点 Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // 当前帧关键点 KeyPoints in current frame
    vector<bool> vbVO, vbMap;          // 跟踪的关键点 Tracked MapPoints in current frame
                                       // vbMap 匹配到地图上一个点
                                       // vbVO 
    int state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);// 对数据上锁====
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;// 切换成 没有图像==

        mIm.copyTo(im);                    // 有update函数从 tracer内拷贝过来======

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;// 类对象 复制过来
            vIniKeys = mvIniKeys;        // 初始关键帧 关键点
            vMatches = mvIniMatches;     // 初始关键帧 关键帧匹配点
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;// 当前关键帧 关键点
            vbVO = mvbVO;   // 跟踪到的
            vbMap = mvbMap;
        }
        else if(mState==Tracking::LOST)// 跟丢了，关键点就没有匹配上===
        {
            vCurrentKeys = mvCurrentKeys;// 只有 当前帧 检测到的关键点
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,CV_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING=====初始化====
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                        cv::Scalar(0,255,0));
            }
        }        
    }
    else if(state==Tracking::OK) //TRACKING  跟踪=====
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i]) // 跟踪到 的关键点=====
            {
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;// 左上方点
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;// 右下方点
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map  匹配到地图上一个点=====
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));// bgr 绿色  正方形
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1);// 内画圆
                    mnTracked++; // 跟踪到的地图点数量====
                }
                else // 跟踪到的上一帧创建的 视觉里程记点 (部分会被删除掉)
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));// bgr  蓝色===
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);// 显示文字

    return imWithInfo;
}


// 显示文字========================================================
void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";// 定位+建图
        else
            s << "LOCALIZATION | ";// 定位
        int nKFs = mpMap->KeyFramesInMap();// 地图中 关键帧数量
        int nMPs = mpMap->MapPointsInMap();// 地图中 地图点数量
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", current Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + current VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);//文字

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());// 图片扩展几行，用来显示文字=========

    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));// 图像拷贝到 带文字框 的图像

    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());// 上次文字区域 清空

    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);// 更新字体=====

}

// 从Track对象中 更新本 类内数据==============================
void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);// 上锁====
    pTracker->mImGray.copyTo(mIm);// 图像
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;// 当前帧 关键帧
    N = mvCurrentKeys.size();// 关键帧数量
    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;// 模式


    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;// 初始化关键帧 关键点
        mvIniMatches=pTracker->mvIniMatches;     // 匹配点====
    }

    else if(pTracker->mLastProcessedState==Tracking::OK)// 跟踪ok
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];//当前帧的地图点
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])// 是否是外点
                {
                    if(pMP->Observations()>0)// 该地图点也被其他 帧观测到，玩儿哦实实在在的地图点
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;// 没有被 观测到，只存在与上一帧=====
                }
            }
        }
    }
    mState=static_cast<int>(pTracker->mLastProcessedState);// 跟踪器状态=========
}

} //namespace ORB_SLAM
