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
    const std::string Const::kTopicRegisteredPC2 = "/camera/depth_registered/points";
    const std::string Const::kTopicPC2 = "pointcloud";
    const std::string Const::kTopicRgb = "rgb";
    const std::string Const::kTopicSegmentation = "segmentation";
    const std::string Const::kTopicDetection = "detection";
    const std::string Const::kTopicLocalization = "localization";
    const std::string Const::kTopicTracking = "tracking";
    
    ros_object_analytics/object_analytics_nodelet/src/
        merger      object_analytics_nodelet::merger
        model       object_analytics_nodelet::model
        segmenter   object_analytics_nodelet::segmenter
                                      2. 点云分割处理器
                                        // 订阅 点云话题消息
                                        sub_ = nh.subscribe(Const::kTopicPC2, 1, &SegmenterNodelet::cbSegment, this);
                                        pub_ = nh.advertise<object_analytics_msgs::ObjectsInBoxes3D>(Const::kTopicSegmentation, 1);
                                        ObjectsInBoxes3D ： x，y，z坐标最大最小值，投影到rgb图像平面上的 ROI框
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
          opencv-3.2 + opencv_contrib-3.2
          $ 下载 opencv-3.2
          $ https://github.com/opencv/opencv/archive/3.2.0.zip 
          $ 解压
          $ cd opencv
          $ 下载 opencv_contrib-3.2
          $ https://codeload.github.com/opencv/opencv_contrib/zip/3.2.0   
          $ 可能需要删除一些 cuda开头的module 和 opencv-3.2本身的module重名了=========
          和 opencv 一起安装
          $ mkdir build
          $ cd build
          $ cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ..
          $ make -j3
          
          
          https://github.com/protocolbuffers/protobuf/releases/download/v3.1.0/protobuf-cpp-3.1.0.tar.gz
    
    
    
