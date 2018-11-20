/**
* This file is part of ORB-SLAM2.
* 地图显示  普通地图点 黑色 参考地图点红色
            关键帧 蓝色   当前帧 绿色
            
*/

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];//关键帧 线长
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];//关键帧线宽
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];// 关键帧连线宽度
    mPointSize = fSettings["Viewer.PointSize"];// 点大小
    mCameraSize = fSettings["Viewer.CameraSize"];// 当前帧 相机线长
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];// 当前帧 相机线宽

	/*
	Viewer.KeyFrameSize: 0.05
	Viewer.KeyFrameLineWidth: 1
	Viewer.GraphLineWidth: 0.9
	Viewer.PointSize:2
	Viewer.CameraSize: 0.08
	Viewer.CameraLineWidth: 3
	Viewer.ViewpointX: 0
	Viewer.ViewpointY: -0.7
	Viewer.ViewpointZ: -1.8
	Viewer.ViewpointF: 500
	*/
}


// 显示点======普通点黑色===参考地图点红色===颜色可修改====
void MapDrawer::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();// 所有地图点  黑色
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();// 参考 地图点 红色===

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());// set有序集合， 查找快!!!!!

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);// 点大小
// 开始添加点===========
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);// 普通地图点 为黑色================rgb=

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)// 所有的地图点=====
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))// 除去不好的 和 参考帧点
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();// 点的时间坐标 位姿
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));// 顶点
    }
// 结束添加点=========
    glEnd();

    glPointSize(mPointSize);// 点大小
// 开始添加点===========
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);// 参考 地图点 显示红色============rgb=======

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;// 除去不好的 
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));// 添加点

    }
// 结束添加点=========
    glEnd();
}

// 显示关键帧================蓝色============================
void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();// 所有关键帧======

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)// 遍例每一个关键帧=====
        {
            KeyFrame* pKF = vpKFs[i];// 关键帧
            cv::Mat Twc = pKF->GetPoseInverse().t();// 帧到世界坐标系====

            glPushMatrix();// 矩阵
            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);//关键帧线宽
            glColor3f(0.0f,0.0f,1.0f);// rgb 蓝色 帧位姿
            glBegin(GL_LINES); // 开始添加线=======

// 相机光心 与 顶点 连线========
            glVertex3f(0,0,0); // 相机光心
            glVertex3f(w,h,z); // 宽 高 深度

            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);

            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);

            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

// 四个顶点之间连线============
            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);

            glEnd();// 画线结束

            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);// rgba  透明度

// 开始画线===============
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph  共视图 ===
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);// 共视图 权重 交大的===
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();

            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree  最小生成树======
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops  闭环帧===连接线======
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId < vpKFs[i]->mnId)// 避免重复画线???
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }
// 结束画线==============
        glEnd();
    }
}


// 显示当前帧 相机位姿========绿色=========================
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;// 当前帧 相机线长
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);// 当前帧 相机线宽
    glColor3f(0.0f,1.0f,0.0f);// 绿色========
// 开始画线=============
    glBegin(GL_LINES);

// 相机光心 与 顶点 连线========
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);

    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);

    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);

    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

// 四个顶点之间连线============
    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
// 结束画线==============
    glEnd();

    glPopMatrix();
}

// 设置当前帧 相机姿======================================
void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

// 获取当前相机位姿，返回 OpenGlMatrix 类型=====
void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

} //namespace ORB_SLAM
