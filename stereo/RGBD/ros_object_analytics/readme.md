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
    
        splitter    object_analytics_nodelet::splitter    
                                      1. RGBD传感器预处理分割器  splitter 
                                      // 订阅  rgbd传感器消息
                                      sub_pc2_ = nh.subscribe(Const::kTopicRegisteredPC2, 1, &SplitterNodelet::cbSplit, this);
                                      // 发布 2d 图像消息
                                      pub_2d_ = nh.advertise<sensor_msgs::Image>(Const::kTopicRgb, 1);
                                      // 发布 3d 点云消息
                                      pub_3d_ = nh.advertise<sensor_msgs::PointCloud2>(Const::kTopicPC2, 1);
                                      
                                      !!!!!! 需要修改为 直接从 图漾rgbd相机获取 彩色图 图 和 点云后 发布出去======
                                      
        segmenter   object_analytics_nodelet::segmenter
                                      2. 点云分割处理器 segmenter
                                        // 订阅 点云话题消息
                                        sub_ = nh.subscribe(Const::kTopicPC2, 1, &SegmenterNodelet::cbSegment, this);
                                        // 发布 点云物体数组
                                        pub_ = nh.advertise<object_analytics_msgs::ObjectsInBoxes3D>(Const::kTopicSegmentation, 1);
                                        ObjectsInBoxes3D ： x，y，z坐标最大最小值，投影到rgb图像平面上的 ROI框
                                        
        merger      object_analytics_nodelet::merger
                                      3. 3d定位器　merger  融合2d检测结果 和 3d分割结果=======
                       using Subscriber2D = message_filters::Subscriber<ObjectsInBoxes>;  // 消息滤波器 订阅 2d边框数组消息
                       using Subscriber3D = message_filters::Subscriber<ObjectsInBoxes3D>;// 消息滤波器 订阅 3d边框数组消息
                       // 订阅  2d检测分割结果 detection
                       sub_2d_ = std::unique_ptr<Subscriber2D>(new Subscriber2D(nh, Const::kTopicDetection, kMsgQueueSize));
                       // 订阅  3d检测分割结果 segmentation
                       sub_3d_ = std::unique_ptr<Subscriber3D>(new Subscriber3D(nh, Const::kTopicSegmentation, kMsgQueueSize));
                       // 同时订阅两个消息的 消息同步器
                       sub_sync_ = std::unique_ptr<ApproximateSynchronizer2D3D>();
                       // 消息同步后，调用回调函数，融合 2d检测结果 和 3d分割结果=======
                       sub_sync_->registerCallback(boost::bind(&MergerNodelet::cbMerge, this, _1, _2));
                       
                       // 发布  3d定位结果　　 localization
                       pub_result_ = nh.advertise<ObjectsInBoxes3D>(Const::kTopicLocalization, 10);
                       
        tracker     object_analytics_nodelet::tracker       
                                     4. 2d 目标跟踪器 tracker
                        // 订阅 RGB图像话题
                        sub_rgb_ = nh.subscribe(Const::kTopicRgb, 6, &TrackingNodelet::rgb_cb, this);
                        // 订阅 2d目标检测结构 话题
                        sub_obj_ = nh.subscribe(Const::kTopicDetection, 6, &TrackingNodelet::obj_cb, this);
                        // 
                        pub_tracking_ = nh.advertise<object_analytics_msgs::TrackedObjects>(Const::kTopicTracking, 10);             
        
        
        model       object_analytics_nodelet::model

## 3. 编译错误信息
    
    a. 缺少 boost/make_unique.hpp 文件
       下载 https://www.boost.org/doc/libs/develop/boost/smart_ptr/make_unique.hpp
       安装到 usr/include/boost/ make_unique.hpp
    b. 缺少 opencv2/tracking.hpp  文件  在 opencv_contrib 中
       安装 opencv_contrib
          opencv-3.2 + opencv_contrib-3.2  安转参考 https://github.com/Ewenwan/opencv3.2_CMake
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
    
    
    
