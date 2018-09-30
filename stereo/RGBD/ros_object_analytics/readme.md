# 使用该RGBD摄像头 + ncnn 目标检测 运行 该目标检测项目

## 1. 安装 object_msgs 包
    Building
    cd ~/catkin_ws/src
    git clone https://github.com/intel/object_msgs
    cd ~/catkin_ws
    catkin_make
    # Installation
    catkin_make install
    source install/setup.bash
    
## 2. 下载 ros_object_analytics 包
    cd ~/catkin_ws/src
    git clone https://github.com/intel/ros_object_analytics.git
    
    分析：
    ros_object_analytics/object_analytics_nodelet/src/
        merger      object_analytics_nodelet::merger
        model       object_analytics_nodelet::model
        segmenter   object_analytics_nodelet::segmenter
        splitter    object_analytics_nodelet::splitter    
                                      1. RGBD传感器预处理分割器 
                                      // 订阅  rgbd传感器消息
                                      sub_pc2_ = nh.subscribe(Const::kTopicRegisteredPC2, 1, &SplitterNodelet::cbSplit, this);
                                      // 发布 2d 图像消息
                                      pub_2d_ = nh.advertise<sensor_msgs::Image>(Const::kTopicRgb, 1);
                                      // 发布 3d 点云消息
                                      pub_3d_ = nh.advertise<sensor_msgs::PointCloud2>(Const::kTopicPC2, 1);
                                      
                                      !!!!!! 需要修改为 直接从 图漾rgbd相机获取 彩色图 图 和 点云后 发布出去======
        tracker     object_analytics_nodelet::tracker
        
## 3. 编译错误信息
    
    a. 缺少 boost/make_unique.hpp 文件
       下载 https://www.boost.org/doc/libs/develop/boost/smart_ptr/make_unique.hpp
       安装到 usr/include/boost/ make_unique.hpp
    b. 缺少 opencv2/tracking.hpp  文件  在 opencv_contrib 中
       安装 opencv_contrib
          $ git clone https://github.com/opencv/opencv.git
          $ 需要 3.4版本
            git branch -a 先查看当前远端分支情况
            git  checkout origin/xxx  选择远端xxx分支
            git branch xxx  创建本地xxx分支
            git checkout xxx  选择新创建的分支就可以了。
          $ cd opencv
          $ git clone https://github.com/opencv/opencv_contrib.git   3.4版本
          $ 可能需要删除一些 cuda开头的module 和 opencv-3.4本身的module重名了=========
          和 opencv 一起安装
          $ mkdir build
          $ cd build
          $ cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ..
          $ make -j3
          
          错误信息：
          /opencv_contrib_master/modules/rgbd/include/opencv2/rgbd/depth.hpp
             add  #include <stdexcept>// runtime_error was not declared
          
          sudo make install 
    
    
    
