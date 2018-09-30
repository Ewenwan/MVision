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
        splitter    object_analytics_nodelet::splitter     1. RGBD传感器预处理分割器 
        tracker     object_analytics_nodelet::tracker
        
## 3. 编译错误信息
    
    a. 缺少 boost/make_unique.hpp 文件
       下载 https://www.boost.org/doc/libs/develop/boost/smart_ptr/make_unique.hpp
       安装到 usr/include/boost/ make_unique.hpp
    b. 缺少 opencv2/tracking.hpp  文件  在 opencv_contrib 中
       安装 opencv_contrib
          $ git clone https://github.com/opencv/opencv.git
          $ cd opencv
          $ git clone https://github.com/opencv/opencv_contrib.git
          和 opencv 一起安装
          $ mkdir build
          $ cd build
          $ cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ..
          $ make -j3
          sudo make install 
    
    
    
