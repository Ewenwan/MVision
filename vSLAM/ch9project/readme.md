# 工程结构 
      1. /bin             存放可执行的二进制文件
      2. /include/myslam  存放slam工程模块的头文件，只要是.h 引用头文件时时
                          需写 include "myslam/xxx.h"不容易和其他库混淆
      3. /src             存放源代码文件   主要是.cpp文件
      4. /test            存放测试用的文件 也是  .cpp文件
      5. /lib             存放编译好的库文件
      6. /config          存放配置文件
      7. /cmake_modules   存放第三方库的cmake文件 例如使用g2o eigen库时


# 数据对象结构
      0.1版本 类
      Frame     帧                Frame::Ptr  frame 
      Camera    相机模型          Camera::Ptr camera_
      MapPoint  特征点/路标点     MapPoint::Ptr map_point 
      Map       管理特征点   保存所有的特征点/路标 和关键帧
      Config    提供配置参数



#  CMakeLists.txt 编译文件
      CMAKE_MINIMUM_REQUIRED( VERSION 2.8 ) # 设定版本
      PROJECT( slam ) # 设定工程名
      SET( CMAKE_CXX_COMPILER "g++") # 设定编译器

      # 设定 可执行 二进制文件 的目录=========
      # 二进制就是可以直接运行的程序
      SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

      # 设定存放 编译出来 的库文件的目录=====
      # 库文件呢，就是为这些二进制提供函数的啦
      SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 
      # 并且把该目录设为 连接目录
      LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

      # 设定头文件目录
      INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include)

      # 增加子文件夹，也就是进入 源代码 文件夹继续构建
      ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src)
      
      # 增加一个可执行的二进制
      ADD_EXECUTABLE( main main.cpp )
      
      # =========================================
      # 增加PCL库的依赖
      FIND_PACKAGE( PCL REQUIRED COMPONENTS common io )

      # 增加opencv的依赖
      FIND_PACKAGE( OpenCV REQUIRED )

      # 添加头文件和库文件
      ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
      INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
      LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

      ADD_EXECUTABLE( generate_pointcloud generatePointCloud.cpp )
      TARGET_LINK_LIBRARIES( generate_pointcloud ${OpenCV_LIBS} 
          ${PCL_LIBRARIES} )
      
      
      # 自检函数库=======
      # 最后，在 src/CMakeLists.txt 中加入以下几行，将 slamBase.cpp 编译成一个库，供将来调用：

      ADD_LIBRARY( slambase slamBase.cpp )
      TARGET_LINK_LIBRARIES( slambase
      ${OpenCV_LIBS} 
      ${PCL_LIBRARIES} )


      
# 2d 点转 3d点  函数
```c
// generatePointCloud.cpp
// https://www.cnblogs.com/gaoxiang12/p/4652478.html

// 部分头文件省略
// 定义点云类型
typedef pcl::PointXYZRGBA PointT; # 点类型
typedef pcl::PointCloud<PointT> PointCloud;  # 点云类型

/*

我们使用OpenCV的imread函数读取图片。在OpenCV2里，图像是以矩阵(cv::MAt)作为基本的数据结构。
Mat结构既可以帮你管理内存、像素信息，还支持一些常见的矩阵运算，是非常方便的结构。
彩色图像含有R,G,B三个通道，每个通道占8个bit（也就是unsigned char），故称为8UC3（8位unsigend char, 3通道）结构。
而深度图则是单通道的图像，每个像素由16个bit组成（也就是C++里的unsigned short），像素的值代表该点离传感器的距离。
通常1000的值代表1米，所以我们把camera_factor设置成1000.
这样，深度图里每个像素点的读数除以1000，就是它离你的真实距离了。

*/


// 相机内参
const double camera_factor = 1000;  // 深度值放大倍数
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

    // 点云变量
    // 使用智能指针，创建一个空点云。这种指针用完会自动释放。
    PointCloud::Ptr cloud ( new PointCloud );
    // 遍历深度图
    // 按照“先列后行”的顺序，遍历了整张深度图。
    for (int m = 0; m < depth.rows; m++)    // 每一行
        for (int n=0; n < depth.cols; n++)  // 每一列
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // 深度图第m行，第n行的数据可以使用depth.ptr<ushort>(m) [n]来获取。
            // 其中，cv::Mat的ptr函数会返回指向该图像第m行数据的头指针。
            // 然后加上位移n后，这个指针指向的数据就是我们需要读取的数据啦。


            
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera_factor;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;
            
            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
```


# 2d 点转 3d点  函数  封装成库

```c
// include/slamBase.h  库头文件
/*************************************************************************
    > File Name: rgbd-slam-tutorial-gx/part III/code/include/slamBase.h
    > Author: xiang gao
    > Mail: gaoxiang12@mails.tsinghua.edu.cn
    > Created Time: 2015年07月18日 星期六 15时14分22秒
    > 说明：rgbd-slam教程所用到的基本函数（C风格）
 ************************************************************************/
# pragma once

// 各种头文件 
// C++标准库
#include <fstream>
#include <vector>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// 类型定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// 相机内参结构===============================
// 把相机参数封装成了一个结构体，
struct CAMERA_INTRINSIC_PARAMETERS 
{ 
    double cx, cy, fx, fy, scale;
};

// 另外还声明了 image2PointCloud 和 point2dTo3d 两个函数
// 函数接口
// image2PonitCloud 将rgb图转换为点云
PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );

// point2dTo3d 将单个点从图像坐标转换为空间坐标
// input: 3维点Point3f (u,v,d)
cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera );

```

```c
// src/slamBase.cpp
/*************************************************************************
    > File Name: src/slamBase.cpp
    > Author: xiang gao
    > Mail: gaoxiang12@mails.tsinghua.edu.cn
    > Implementation of slamBase.h
    > Created Time: 2015年07月18日 星期六 15时31分49秒
 ************************************************************************/

#include "slamBase.h"
// image2PonitCloud 将rgb图转 换为 点云====================
PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;
            
            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}
// point2dTo3d 将单个点从图像坐标转换为空间坐标
// input: 3维点Point3f (u,v,d)
cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}



```

      # 自检函数库=======
      # 最后，在 src/CMakeLists.txt 中加入以下几行，将 slamBase.cpp 编译成一个库，供将来调用：

      ADD_LIBRARY( slambase slamBase.cpp )
      TARGET_LINK_LIBRARIES( slambase
      ${OpenCV_LIBS} 
      ${PCL_LIBRARIES} )



# 图像配准 数学部分   3d-3d配准
      用基于特征的方法（feature-based）或直接的方法（direct method）来解。
      虽说直接法已经有了一定的发展，但目前主流的方法还是基于特征点的方式。
      在后者的方法中，首先你需要知道图像里的“特征”，以及这些特征的一一对应关系。

      假设我们有两个帧：F1和F2. 并且，我们获得了两组一一对应的 特征点：
            P={p1,p2,…,pN}∈F1
            Q={q1,q2,…,qN}∈F2
       其中p和q都是 R3 中的点。

      我们的目的是求出一个旋转矩阵R和位移矢量t，使得：
        ∀i, pi = R*qi + t

      然而实际当中由于误差的存在，等号基本是不可能的。所以我们通过最小化一个误差来求解R,t:
      　min R,t ∑i=1/N * ∥pi−(R*qi + t)∥2
　　   这个问题可以用经典的ICP算法求解。其核心是奇异值分解(SVD)。
      我们将调用OpenCV中的函数求解此问题，
      
      那么从这个数学问题上来讲，我们的关键就是要获取一组一一对应的空间点，
      这可以通过图像的特征匹配来完成。　　
      提示：由于OpenCV中没有提供ICP，我们在实现中使用PnP进行求解。 2d-3d
# 配准编程
```c
// detectFeatures.cpp 
/*************************************************************************
	> File Name: detectFeatures.cpp
	> Author: xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn
    > 特征提取与匹配
	> Created Time: 2015年07月18日 星期六 16时00分21秒
 ************************************************************************/

#include<iostream>
#include "slamBase.h"
using namespace std;

// OpenCV 特征检测模块
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
#include <opencv2/calib3d/calib3d.hpp>

int main( int argc, char** argv )
{
    // 声明并从data文件夹里读取两个rgb与深度图
    cv::Mat rgb1 = cv::imread( "./data/rgb1.png");
    cv::Mat rgb2 = cv::imread( "./data/rgb2.png");
    cv::Mat depth1 = cv::imread( "./data/depth1.png", -1);
    cv::Mat depth2 = cv::imread( "./data/depth2.png", -1);

    // 声明特征提取器 与 描述子提取器
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;

    // 构建提取器，默认两者都为 ORB
    
    // 如果使用 sift, surf ，之前要初始化nonfree模块=========
    // cv::initModule_nonfree();
    // _detector = cv::FeatureDetector::create( "SIFT" );
    // _descriptor = cv::DescriptorExtractor::create( "SIFT" );
    
    detector = cv::FeatureDetector::create("ORB");
    descriptor = cv::DescriptorExtractor::create("ORB");
    
//  使用 _detector->detect()函数提取关键点==============================
    // 关键点是一种cv::KeyPoint的类型。
    // 带有 Point2f pt 这个成员变量，指这个关键点的像素坐标。
    
    // kp1[i].pt 获取 这个关键点的像素坐标 (u，v) ==================
    
    // 此外，有的关键点还有半径、角度等参数，画在图里就会像一个个的圆一样。
    vector< cv::KeyPoint > kp1, kp2; // 关键点
    detector->detect( rgb1, kp1 );   // 提取关键点
    detector->detect( rgb2, kp2 );

    cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
    
    // 可视化， 显示关键点
    cv::Mat imgShow;
    cv::drawKeypoints( rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imshow( "keypoints", imgShow );
    cv::imwrite( "./data/keypoints.png", imgShow );
    cv::waitKey(0); //暂停等待一个按键
   
    // 计算描述子===================================================
    // 在 keypoint 上计算描述子。
    // 描述子是一个cv::Mat的矩阵结构，
    // 它的每一行代表一个对应于Keypoint的特征向量。
    // 当两个keypoint的描述子越相似，说明这两个关键点也就越相似。
    // 我们正是通过这种相似性来检测图像之间的运动的。
    cv::Mat desp1, desp2;
    descriptor->compute( rgb1, kp1, desp1 );
    descriptor->compute( rgb2, kp2, desp2 );

    // 匹配描述子===================================================
    // 对上述的描述子进行匹配。
    // 在OpenCV中，你需要选择一个匹配算法，
    // 例如粗暴式（bruteforce），近似最近邻（Fast Library for Approximate Nearest Neighbour, FLANN）等等。
    // 这里我们构建一个FLANN的匹配算法：
    vector< cv::DMatch > matches; 
    // cv::BFMatcher matcher;      // 暴力匹配，穷举
    cv::FlannBasedMatcher matcher; // 近似最近邻
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

// 匹配完成后，算法会返回一些 DMatch 结构。该结构含有以下几个成员：
//    queryIdx 源特征描述子的索引（也就是第一张图像，第一个参数代表的desp1）。
//    trainIdx 目标特征描述子的索引（第二个图像，第二个参数代表的desp2）
//    distance 匹配距离，越大表示匹配越差。  matches[i].distance
// matches.size() 总数


//　　有了匹配后，可以用drawMatch函数画出匹配的结果：

    // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
    cv::imwrite( "./data/matches.png", imgMatches );
    cv::waitKey( 0 );

    // 筛选匹配，把距离太大的去掉
    // 这里使用的准则是去掉大于四倍最小距离的匹配
    // 筛选的准则是：去掉大于最小距离四倍的匹配。====================================
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }
    cout<<"min dis = "<<minDis<<endl;// 最好的匹配===============

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 10*minDis)
            goodMatches.push_back( matches[i] );// 筛选出来的 剩下的较好的匹配
    }

    // 显示 good matches
    cout<<"good matches="<<goodMatches.size()<<endl;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
    cv::imshow( "good matches", imgMatches );
    cv::imwrite( "./data/good_matches.png", imgMatches );
    cv::waitKey(0);

    // 计算图像间的运动关系
    // 关键函数：cv::solvePnPRansac()
    // 为调用此函数准备必要的参数
    
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;// desp1 的2d点 利用深度值 转换成 3d点
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 325.5;
    C.cy = 253.5;
    C.fx = 518.0;
    C.fy = 519.0;
    C.scale = 1000.0;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;      // 2d点==============
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！！！！！！！！
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];// 从深度图 取得深度===
        if (d == 0)
            continue;// 跳过深度值异常的点=============
            
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );// 图像2 关键点对应的 2d像素点

        // 将(u,v,d)转成(x,y,z)=======================
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );// 图像1 关键点对应 2d像素点 对应的 3d点
    }

// 相机内参数矩阵K ===============================
    double camera_matrix_data[3][3] =
    {
        {C.fx, 0,    C.cx},
        {0,    C.fy, C.cy},
        {0,    0,    1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );// 8字节
    cv::Mat rvec, tvec, inliers;
    // 求解pnp            3d点     2d点  相机内参数矩阵K         旋转矩阵rvec 平移向量tvec
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
    
// 这个就叫做“幸福的家庭都是相似的，不幸的家庭各有各的不幸”吧。
// 你这样理解也可以。ransac适用于数据噪声比较大的场合

    cout<<"inliers: "<<inliers.rows<<endl; // ransac 随机采样一致性 得到的内点数量
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    // 画出inliers匹配 
    vector< cv::DMatch > matchesShow; // 好的匹配
    for (size_t i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );// inliers 第i行的地一个参数为 匹配点id
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::imwrite( "./data/inliers.png", imgMatches );
    cv::waitKey( 0 );

    return 0;
}


```


# 配准程序 放入库 
```c
// 装进slamBase库中， 
// 在 include/slamBase.h  扩展 以下代码

// FRAME 帧结构=============== 结构体
struct FRAME
{
    cv::Mat rgb, depth; // 该帧对应的 彩色图 与深度图
    cv::Mat desp;       // 特征描述子 集 一行对应一个关键点
    vector<cv::KeyPoint> kp; // 关键点 集  kp[i].pt 是关键点对应的像素坐标
};

// PnP 结果 2d-3d 配置结果===== 结构体
struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    int inliers;
};

// computeKeyPointsAndDesp 同时提取关键点与特征描述子======================
void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor );

// estimateMotion 2d-3d pnp配准 计算两个帧之间的运动=======================
// 输入：帧1和帧2, 相机内参
RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera );


```







