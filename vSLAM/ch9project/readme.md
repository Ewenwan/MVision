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

# RGBD SLAM 工程 记录 
[rgbdslam_v2 参考](https://github.com/Ewenwan/rgbdslam_v2)

[高翔博士 博客](https://www.cnblogs.com/gaoxiang12/p/4669490.html)

[高翔博士代码](https://github.com/gaoxiang12/rgbd-slam-tutorial-gx)
      
      
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
    // 求解pnp            3d点     2d点  相机内参数矩阵K          旋转向量 rvec 平移向量tvec
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

// 我们把关键帧和PnP的结果都封成了结构体，以便将来别的程序调用==========

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
    int inliers; // 内点数量=====!!!!
};

// computeKeyPointsAndDesp 同时提取关键点与特征描述子========引用传递==============
void computeKeyPointsAndDesp( FRAME & frame, string detector, string descriptor );

// estimateMotion 2d-3d pnp配准 计算两个帧之间的运动==========引用传递==============
// 输入：帧1和帧2, 相机内参
RESULT_OF_PNP estimateMotion( FRAME & frame1, FRAME & frame2, CAMERA_INTRINSIC_PARAMETERS& camera );


```

```c
// 提取关键点与特征描述子函数 2d-3d-pnp配准函数 src/slamBase.cpp=========================

// computeKeyPointsAndDesp 同时提取关键点与特征描述子============引用传递============== 
void computeKeyPointsAndDesp( FRAME & frame, string detector, string descriptor )
{
    cv::Ptr<cv::FeatureDetector> _detector;        // 关键点检测
    cv::Ptr<cv::DescriptorExtractor> _descriptor;  // 描述子计算

    cv::initModule_nonfree(); // 如果使用 SIFI / SURF 的话========
    
    _detector = cv::FeatureDetector::create( detector.c_str() );
    _descriptor = cv::DescriptorExtractor::create( descriptor.c_str() );

    if (!_detector || !_descriptor)
    {
        cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
        return;
    }

    _detector->detect( frame.rgb, frame.kp ); // 检测关键点
    _descriptor->compute( frame.rgb, frame.kp, frame.desp );// 计算描述子

    return;
}

// estimateMotion 计算两个帧之间的运动==========================================================
// 输入：帧1和帧2
// 输出：rvec 和 tvec
RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    static ParameterReader pd;   // // 好关键点阈值 参数 读取==============
    vector< cv::DMatch > matches;// 匹配点对
    cv::FlannBasedMatcher matcher;// 快速最近邻 匹配器============
    matcher.match( frame1.desp, frame2.desp, matches );// 对两个关键帧的关键点 进行匹配
   
    cout<<"find total "<<matches.size()<<" matches."<<endl;
// 初步筛选 好的匹配点对==========================
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    
    double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );// 好关键点阈值 读取
    
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance; // 最好的匹配点对  对应的匹配距离
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] ); // 筛选下来的 好的 匹配点对
    }

    cout<<"good matches: "<<goodMatches.size()<<endl;
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;  // 2d 关键点对应的像素点 + 对应的深度距离 根据相机参数 转换得到
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;// 2d 关键点对应的像素点
    
// 从匹配点对 获取 2d-3d点对 ===========================
    for (size_t i=0; i<goodMatches.size(); i++) 
    {
        // query 是第一个, train 是第二个 得到的是 关键点的id
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;// .pt 获取关键点对应的 像素点
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];// y行，x列
        if (d == 0)
            continue; // 深度值不好 跳过
        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d ); // 2d 关键点对应的像素点 + 对应的深度距离
        cv::Point3f pd = point2dTo3d( pt, camera );// 根据相机参数 转换得到 3d点
        pts_obj.push_back( pd );
	
	
	pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );// 后一帧的 2d像素点

    }
    
// 相机内参数矩阵 K =========================
    double camera_matrix_data[3][3] = 
    {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    cout<<"solving pnp"<<endl;
    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp            3d点     2d点  相机内参数矩阵K         旋转向量 rvec 平移向量tvec
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
    // 旋转向量形式 3×1 rvec
// 这个就叫做“幸福的家庭都是相似的，不幸的家庭各有各的不幸”吧。
// 你这样理解也可以。ransac适用于数据噪声比较大的场合

    RESULT_OF_PNP result;  // 2D-3D匹配结果
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows; //   内点数量=====!!!!

    return result; 返回
}


```


# 文件参数 读取类
	此外，我们还实现了一个简单的参数读取类。
	这个类读取一个参数的文本文件，能够以关键字的形式提供文本文件中的变量。
```c
// 装进slamBase库中， 
// 在 include/slamBase.h  扩展 以下代码

// 参数读取类
class ParameterReader
{
public:
    ParameterReader( string filename="./parameters.txt" )
    {
        ifstream fin( filename.c_str() );
        if (!fin)
        {
            cerr<<"parameter file does not exist."<<endl;
            return;
        }
        while(!fin.eof())// 知道文件结尾
        {
            string str;
            getline( fin, str );// 每一行======
            if (str[0] == '#')// [0]是开头的第一个字符
            {
                // 以‘＃’开头的是注释
                continue;
            }

            int pos = str.find("="); // 变量赋值等号 =  前后 无空格=====
            if (pos == -1) 
                continue;// 没找到 =
		
            string key = str.substr( 0, pos ); // 变量名字====
            string value = str.substr( pos+1, str.length() );// 参数值 字符串
            data[key] = value; // 装入字典========

            if ( !fin.good() )
                break;
        }
    }
    
    string getData( string key ) // 按关键字 在 字典中查找========
    {
        map<string, string>::iterator iter = data.find(key);// 二叉树查找 log(n)
        if (iter == data.end())
        {
            cerr<<"Parameter name "<<key<<" not found!"<<endl;
            return string("NOT_FOUND");
        }
        return iter->second; // 返回对应关键字对应 的 值  iter->first 为 key  iter->second 为值
    }
    
public:
    map<string, string> data; // 解析得到的参数字典
    
};

// 示例参数
它读的参数文件是长这个样子的：

# 这是一个参数文件
# 去你妹的yaml! 我再也不用yaml了！简简单单多好！
# 等号前后不能有空格
# part 4 里定义的参数

detector=ORB
descriptor=ORB
good_match_threshold=4

# camera
camera.cx=325.5;
camera.cy=253.5;
camera.fx=518.0;
camera.fy=519.0;
camera.scale=1000.0;

# 如果我们想更改特征类型，就只需在parameters.txt文件里进行修改，不必编译源代码了。
# 这对接下去的各种调试都会很有帮助。
```


# 点云拼接

	点云的拼接，实质上是对点云做变换的过程。这个变换往往是用变换矩阵(transform matrix)来描述的：
	T=[R t
	   O 1]∈R4×4
	该矩阵的左上部分 R 是一个3×3的旋转矩阵，它是一个正交阵。
	右上部分 t 是3×1的位移矢量。
	左下O是3×1的 !!!缩放矢量!!!!，在SLAM中通常取成0，
	因为环境里的东西不太可能突然变大变小（又没有缩小灯）。
	
	右下角是个1. 这样的一个阵可以对点或者其他东西进行齐次变换。

	[X1        [X2
	 Y1         Y2
	 Z1    = T⋅ Z2 
	 1]         1]   
         由于变换矩阵t 结合了 旋转R 和 平移t，是一种较为经济实用的表达方式。
	 它在机器人和许多三维空间相关的科学中都有广泛的应用。
	 PCL里提供了点云的变换函数，只要给定了变换矩阵，就能对移动整个点云：
	 
	pcl::transformPointCloud( input, output, T );
	
	OpenCV认为旋转矩阵R，虽然有3×3 那么大，自由变量却只有三个，不够节省空间。
	所以在OpenCV里使用了一个向量来表达旋转。
	向量的方向是旋转轴，大小则是转过的弧度.
        我们先用 罗德里格斯变换（Rodrigues）将旋转向量转换为矩阵，然后“组装”成变换矩阵。
	代码如下：

```c
// src/jointPointCloud.cpp===============================
/*****************************************************
	> File Name: src/jointPointCloud.cpp
	> Author: Xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn 
	> Created Time: 2015年07月22日 星期三 20时46分08秒
 **********************************************/

#include<iostream>
using namespace std;

#include "slamBase.h"

#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h> // 点云转换
#include <pcl/visualization/cloud_viewer.h> // 点云显示

// Eigen !
#include <Eigen/Core>
#include <Eigen/Geometry>

int main( int argc, char** argv )
{
// 参数读取器， 请见include/slamBase.h
    ParameterReader pd;
    // 声明两个帧，FRAME结构请见include/slamBase.h
    FRAME frame1, frame2;
   //本节要拼合data中的两对图像
    //读取图像==============================
    frame1.rgb = cv::imread( "./data/rgb1.png" );
    frame1.depth = cv::imread( "./data/depth1.png", -1);// 
    frame2.rgb = cv::imread( "./data/rgb2.png" );
    frame2.depth = cv::imread( "./data/depth2.png", -1 );

    // 提取特征并计算描述子====================
    cout<<"extracting features"<<endl;
    string detecter = pd.getData( "detector" );     // 参数读取 特征检测器
    string descriptor = pd.getData( "descriptor" ); // 描述子计算器
    // 计算 特征点和描述子=
    computeKeyPointsAndDesp( frame1, detecter, descriptor );
    computeKeyPointsAndDesp( frame2, detecter, descriptor );

    // 相机内参=========================================
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str()); // 参数读取 相机参数 字符串转换成 浮点型
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );

    cout<<"solving pnp"<<endl;
    // 求解 pnp  2d-3d 配准估计变换======================================
    RESULT_OF_PNP result = estimateMotion( frame1, frame2, camera );

    cout<<result.rvec<<endl<<result.tvec<<endl;

    // 处理result
    // 将 旋转向量 转化为 旋转矩阵
    cv::Mat R;
    cv::Rodrigues( result.rvec, R ); // 旋转向量  罗德里格斯变换 转换为 旋转矩阵
    Eigen::Matrix3d r;// 3×3矩阵
    cv::cv2eigen(R, r);
  
    // 将平移向量 和 旋转矩阵 转换 成 变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);// 旋转矩阵
    cout<<"translation"<<endl; // 平移向量============
    Eigen::Translation<double,3> trans(result.tvec.at<double>(0,0), 
                                       result.tvec.at<double>(0,1),
				       result.tvec.at<double>(0,2));// 多余??????=========
    T = angle;// 旋转矩阵 赋值 给 变换矩阵 T 
    T(0,3) = result.tvec.at<double>(0,0); // 添加 平移部分
    T(1,3) = result.tvec.at<double>(0,1); 
    T(2,3) = result.tvec.at<double>(0,2);

    // 转换点云
    cout<<"converting image to clouds"<<endl;
    PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, camera ); // 帧1 对应的 点云
    PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera ); // 针2 对应的 点云

    // 合并点云
    cout<<"combining clouds"<<endl;
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *cloud1, *output, T.matrix() );// F1下的 c1点云  T*c1 --> F2下 
    
    *output += *cloud2;// 在 F2下 加和 两部分 点云 ========
    pcl::io::savePCDFile("data/result.pcd", *output);
    cout<<"Final result saved."<<endl;

    pcl::visualization::CloudViewer viewer( "viewer" );
    viewer.showCloud( output );
    while( !viewer.wasStopped() )
    {
        
    }
    return 0;
}

// 至此，我们已经实现了一个只有两帧的SLAM程序。然而，也许你还不知道，这已经是一个视觉里程计(Visual Odometry)啦！
// 只要不断地把进来的数据与上一帧对比，就可以得到完整的运动轨迹以及地图了呢！

// 以两两匹配为基础的里程计有明显的累积误差，我们需要通过回环检测来消除它。这也是我们后面几讲的主要内容啦！
// 我们先讲讲关键帧的处理，因为把每个图像都放进地图，会导致地图规模增长地太快，所以需要关键帧技术。
// 然后呢，我们要做一个SLAM后端，就要用到g2o啦！

```


# Visual Odometry (视觉里程计)
[视频流数据 RGB+Depth 400M+ 取自nyuv2数据集](https://yun.baidu.com/s/1i33uvw5)

	http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html 

	这可是一个国际上认可的，相当有名的数据集哦。如果你想要跑自己的数据，当然也可以，不过需要你进行一些预处理啦。


	实际上和滤波器很像，通过不断的两两匹配，估计机器人当前的位姿，过去的就给丢弃了。
	这个思路比较简单，实际当中也比较有效，能够保证局部运动的正确性。


> 旋转向量 和平移向量 变换成 变换矩阵T 放入库
```c
//  src/slamBase.cpp
// cvMat2Eigen
// 旋转向量 rvec 和 平移向量 tvec 变换成 变换矩阵T===========================
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec )
{
    cv::Mat R;
    // 旋转向量 rvec 变成 旋转矩阵==========
    cv::Rodrigues( rvec, R );
    // cv 3×3矩阵 转换成 Eigen 3×3 矩阵====
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ ) 
            r(i,j) = R.at<double>(i,j);// 8×8字节 double
  
    // 将平移向量 和 旋转矩阵 转换成 变换矩阵 T
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();// 单位阵

    Eigen::AngleAxisd angle(r);// 旋转矩阵 >>> Eigen 旋转轴  
    T = angle;// 旋转轴 >>> 变换矩阵
    T(0,3) = tvec.at<double>(0,0);  // 附加上 平移向量
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);
    return T;
}


```

> 前后两帧点云合并
```c

// joinPointCloud 
// 输入：原始点云，新来的帧 以及 它的位姿
// 输出：将新来帧加到原始帧后的图像
PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, // 原始点云
                                FRAME& newFrame,          // 新来的帧
				Eigen::Isometry3d T,      // 它的位姿，相对 原始点云的位姿
				CAMERA_INTRINSIC_PARAMETERS& camera ) // 相机参数
{
    // 新来的帧 根据 RGB 和 深度图 产生 一帧点云 =======
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera );

    // 合并点云
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );// 怎么是前面的点云 变换到 当前帧 下
    *newCloud += *output; // 当前帧 点云 和变换的点云 加和

    // Voxel grid 滤波降采样
    static pcl::VoxelGrid<PointT> voxel;// 静态变量  体素格下采样，只会有一个 变量实体======================
    static ParameterReader pd;          // 静态变量 文件参数读取器 
    double gridsize = atof( pd.getData("voxel_grid").c_str() );// 体素格精度 
    voxel.setLeafSize( gridsize, gridsize, gridsize );// 设置体素格子 大小
    voxel.setInputCloud( newCloud );// 输入点云
    PointCloud::Ptr tmp( new PointCloud() );// 临时点云
    voxel.filter( *tmp );// 滤波输出点云
    return tmp;
}


```
