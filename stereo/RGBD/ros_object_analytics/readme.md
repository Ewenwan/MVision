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
        tracker     object_analytics_nodelet::tracker
        
## 3. 编译错误信息
    
    a. 缺少 boost/make_unique.hpp 文件
       下载 https://www.boost.org/doc/libs/develop/boost/smart_ptr/make_unique.hpp
       安装到 usr/include/boost/ make_unique.hpp
    b. 缺少 opencv2/tracking.hpp  文件   这是opencv3.2以上才有的
       更新opencv 到 3.2以上
    
    
    
    
