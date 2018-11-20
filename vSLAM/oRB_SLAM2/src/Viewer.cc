/**
* This file is part of ORB-SLAM2.
* 可视化器，使用地图显示器(胖果林) + 帧显示器(opencv)======
  帧率、图像尺寸 胖果林显示地图点 当前帧 关键帧 cv显示特征点图像 菜单响应
*/

#include "Viewer.h"
#include <pangolin/pangolin.h>

#include <mutex>

namespace ORB_SLAM2
{

Viewer::Viewer(System* pSystem, 
               FrameDrawer *pFrameDrawer, 
               MapDrawer *pMapDrawer, 
               Tracking *pTracking, 
               const string &strSettingPath):
               mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),
               mpMapDrawer(pMapDrawer), mpTracker(pTracking),
               mbFinishRequested(false), mbFinished(true), 
               mbStopped(true), mbStopRequested(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];// 帧率
    if(fps<1)
        fps=30;
    mT = 1e3/fps;// ms 帧率倒数

    mImageWidth = fSettings["Camera.width"];
    mImageHeight = fSettings["Camera.height"];
    if(mImageWidth<1 || mImageHeight<1)
    {// 图像尺寸
        mImageWidth = 640;
        mImageHeight = 480;
    }
// 视点位置=======
    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
/*
	Viewer.ViewpointY: -0.7
	Viewer.ViewpointZ: -1.8
	Viewer.ViewpointF: 500
*/
}


// 可视化主线程 函数=================
void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

// 1. 窗口设置 pangolin 胖果林 创建 地图 显示窗口=====1024×768=====
    pangolin::CreateWindowAndBind("地图显示",1024,768);// 窗口名字和窗口大小=====

// 2. 混合颜色设置======
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
// 检查，当前像素前面是否有别的像素，如果别的像素挡道了它，那它就不会绘制，
// 也就是说，OpenGL就只绘制最前面的一层。

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND); // 打开混合
// 基于源像素Alpha通道值的半透明混合函数
// 透过红色的玻璃去看绿色的物体，那么可以先绘制绿色的物体，再绘制红色玻璃。
// 在绘制红色玻璃的时候，利用“混合”功能，把将要绘制上去的红色和原来的绿色进行混合，
// 于是得到一种新的颜色，看上去就好像玻璃是半透明的。
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);// 颜色混合

// 3. 窗口菜单设置=============
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
// 菜单栏====
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);// 地图视角跟随相机动  默认不勾选
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);// 显示地图点
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);// 显示关键帧
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);// 显示关键帧 连线
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);// 仅定位模式 默认不勾选

    pangolin::Var<bool> menuReset("menu.Reset",false,false);// 重置  单行单按钮

    // Define Camera Render Object (for view / scene browsing)
// 窗口视角  ========
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();


// OPENCV 显示 当前帧==============================
    cv::namedWindow("当前帧+关键点");

    bool bFollow = true;
    bool bLocalizationMode = false;

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);// 当前帧 位姿 OpenGL Matrix

        if(menuFollowCamera && bFollow)//menuFollowCamera 为菜单获取的值
        {
            s_cam.Follow(Twc); // 视角跟随 相机位姿===
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;// 防止一直循环====
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode(); // 仅定位模式
            bLocalizationMode = true;// 防止一直循环====
        }
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();// 定位+建图
            bLocalizationMode = false;// 防止一直循环====
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);

        mpMapDrawer->DrawCurrentCamera(Twc);// 绘制当前帧
        if(menuShowKeyFrames || menuShowGraph)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph);// 绘制关键帧 及其之间的连线
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();// 显示地图点

        pangolin::FinishFrame(); // 胖果林完成显示=================

        cv::Mat im = mpFrameDrawer->DrawFrame(); // 返回关键帧，带有 关键点========
        cv::imshow("当前帧+关键点",im);
        cv::waitKey(mT);

        if(menuReset) // 重置====
        {
            menuShowGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->Reset();
            menuReset = false;
        }

        if(Stop())//停止=====
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())// 检查是否停止====
            break;
    }

    SetFinish();
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

}
