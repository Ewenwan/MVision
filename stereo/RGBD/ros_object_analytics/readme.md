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
                        sub_rgb_ = nh.subscribe(Const::kTopicRgb, 6, &TrackingNodelet::rgb_cb, this); // sensor_msgs::Image
                        // 订阅 2d目标检测结构 话题
                        sub_obj_ = nh.subscribe(Const::kTopicDetection, 6, &TrackingNodelet::obj_cb, this);
                                                                                                   // object_msgs::ObjectsInBoxes
                        // 发布 2d目标跟踪结果
                        pub_tracking_ = nh.advertise<object_analytics_msgs::TrackedObjects>(Const::kTopicTracking, 10);             
                            // object_analytics_msgs::TrackedObjects  目标类别id + roi框 数组
                        
                        /ros_object_analytics/object_analytics_nodelet/src/tracker/tracking.cpp 49行
                        tracker 对象创建可能会出现编译错误，老版本创建tracker对象时需要传入类型，新版本之间指定 特点类型的tracker
                        tracker_ = cv::TrackerMIL::create();// 新版本 >>>> tracker_ = cv::Tracker::create("MIL");
                        
                   检测结果的每一个roi会用来初始化一个跟踪器。之后会跟踪后面的每一帧，直到下一个检测结果来到。
                   [detection, tracking, tracking, tracking, ..., tracking] [detection, tracking, tracking, tracking, ..., tracking]
                   
        model       object_analytics_nodelet::model
                     5. 模型类
                        11. object_analytics_nodelet::model::Object2D    
                            const sensor_msgs::RegionOfInterest roi_;// 物体边框
                            const object_msgs::Object object_;       // 物体名称 + 概率
                            
                        22. object_analytics_nodelet::model::Object3D  
                            sensor_msgs::RegionOfInterest roi_;      // 2点云团对应的图像的 roi
                            geometry_msgs::Point32 min_;             // 三个坐标轴 最小的三个量(用于3d(长方体))
                            geometry_msgs::Point32 max_;             // 三个坐标轴 最大的三个量
                            
                        33. object_analytics_nodelet::model::ObjectUtils
                            PointXYZPixel PCL新定义点类型   3d点坐标+2d的像素坐标值 3d-2d点对    
                            ObjectUtils::copyPointCloud(); // 指定index3d点 pcl::PointXYZRGBA >>>> PointXYZPixel
                                      pcl::copyPointCloud(*original, indices, *dest);// 拷贝 3d点坐标
                                      uint32_t width = original->width;// 相当于图像宽度 640 × 480
                                      for (uint32_t i = 0; i < indices.size(); i++)// 所有指定3d点 indices为 3d点序列id
                                      {
                                        dest->points[i].pixel_x = indices[i] % width;// 像素 列坐标
                                        dest->points[i].pixel_y = indices[i] / width;// 像素 行坐标
                                      }
                            ObjectUtils::getProjectedROI( PointXYZPixel ); // 获取点云对应2d像素坐标集合的 包围框 ROI
                            点云团 x、y、z 值得最大值和最小值
                            ObjectUtils::getMatch(): // 计算 两个边框得相似度 = 交并比 * 边框中心距离 / 平均长宽计算的面积
                            ObjectUtils::findMaxIntersectionRelationships(); // 输入 2d 目标检测框，3d 点云对应得2d框
                                                                             // 为每一个2d框 找一个最相似的 3d点云2d框(对应点云团)
                                      a. 遍历 每一个2d物体对应的2d框
                                           b. 遍历   每一个3d物体对应的2d框
                                                c. 调用 getMatch()，计算两边框d相似度
                                                d. 记录最大相似读，对应的 3d物体对应的2d框
                                                e. 记录 pair<Object2D, Object3D> 配对关系
                                                
    object_analytics_visualization 可视化节点

      6. 3d定位可视化　visualization3d　localization
         输入: 2d检测结果 "detection_topic" default="/object_analytics/detection" 
         输入: 3d检测结果 "localization_topic" default="/object_analytics/localization" 
         输入: 2d跟踪结果 "tracking_topic" default="/object_analytics/tracking" 
         输出: rviz可视化话题　"output_topic" default="localization_rviz" 

      7. 2d跟踪可视化　visualization2d 
         输入: 2d检测结果 "detection_topic" default="/object_analytics/detection" 
         输入: 2d跟踪结果 "tracking_topic" default="/object_analytics/tracking" 
         输入: 2d图像     "image_topic" default="/object_analytics/rgb" 
         输出: rviz可视化话题　""output_topic" default="tracking_rviz"    

                            
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
          
          
          以上可能都有问题  安装opencv-3.1.0 + opencv_contrib-3.1.0
           opencv-3.1.0 下载         hhttps://github.com/opencv/opencv/tree/3.1.0
           opencv_contrib-3.1.0 下载 https://github.com/opencv/opencv_contrib/tree/3.1.0 放到opencv-3.1.0/下 并改名为 opencv_contrib
            安装依赖项:
            sudo apt-get install build-essential libgtk2.0-dev libvtk5-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev libtbb-dev 

            编译依赖 sudo apt-get install build-essential
            必须     sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev     
            可选     sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

            安装 lapacke : sudo apt-get install liblapacke-dev checkinstall
            编译: 
            cd opencv-3.1.0
            mkdir build
            cd build 
            cmake -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ ..
            make -j2
            sudo make install
            
            下在未成功的文件 可以从这里下载 https://github.com/Ewenwan/opencv3.2_CMake
    
##  4. 需要修改的地方
     A.  RGBD传感器预处理分割器  splitter 
         这里需要使用 图漾RGBD API 获取校正后的rgbd 图像 和 和rgb图像陪准的点云 
         const std::string Const::kTopicPC2 = "pointcloud"; // 点云话题  发布数据类型:  sensor_msgs::PointCloud2  与pcl点云个是不一致
                    Header header   # 时间戳 
                    uint32 height   # 点云的2d 有序结构(来在与图像)， 若无序， height = 1
                    uint32 width
                    PointField[] fields  # 存储的数据结构
                    bool    is_bigendian # 大端模式??
                    uint32  point_step   # 一个点 的 字节宽度
                    uint32  row_step     # 一行的  字节宽度
                    uint8[] data         # Actual point data, size is (row_step*height)
                    bool is_dense        # 是否为稠密点云，稠密点云，无不合格的数据
         const std::string Const::kTopicRgb = "rgb";        // rgb话题   发布数据类型:  sensor_msgs::Image
     
     B. 原来的 目标检测节点 需要替换
        a. openCL + YOLO_V2 
        b. NCS + MobileNetSSD
        c. ncnn + mobileNetv2SsdLite 使用 ncnn框架实现 目标检测功能，这里的检测频率不需要过快，后面有Tracker节点会对目标检测结果进行跟踪
            这里有两个参数， 一个是目标就爱你测框得置信度  和 
            (检测结果的 抽取宽度,实际为跟踪次数阈值，被跟踪次数长度，年龄？？？，跟踪的次数越多，可能就越不准确===)， 
            好像都会被传递到 Tracker 节点
            其实这里可以直接在 Detection 节点 直接抽取，
            因为相机拍摄后，又输出点云，帧率不是很高，5~10帧，目标检测帧率3~5帧，所以目标检测节点可以采用采样检测。
            这里 Tracker 节点可能也需要修改 =======
            
        
        
## 5. 修改记录

### A. 首先添加一个新的节点 detector
    ros_object_analytics/object_analytics_nodelet/include/object_analytics_nodelet/detector/detector.h
    ros_object_analytics/object_analytics_nodelet/include/object_analytics_nodelet/detector/detector_nodelet.h
    ros_object_analytics/object_analytics_nodelet/include/object_analytics_nodelet/detector/ncnn/lib/libncnn.a
    ros_object_analytics/object_analytics_nodelet/include/object_analytics_nodelet/detector/ncnn/include/*.h
    ros_object_analytics/object_analytics_nodelet/src/detector/detector_nodelet.cpp
    ros_object_analytics/object_analytics_nodelet/src/detector/detector.cpp
    
### B. 添加一个新 项目，获取 图漾相机 XYZRGBD点云数据 后发送出去
    ty api 依赖opencv3.x  
    而 cv_bridge 在 indigo版本下，默认为 2.4版本，编译正确，但是运行 出错，core dump，核心已转储
    需要源码安转 cv_bridge 
    https://github.com/ros-perception/vision_opencv/tree/indigo
    
    cd catkin_ws/src
    catkin_create_pkg ty_rgbd_node roscpp std_msgs sensor_msgs pcl_conversions image_transport nodelet
    
    参考 https://github.com/ros-perception/image_pipeline/blob/indigo/stereo_image_proc/src/libstereo_image_proc/processor.cpp
    
    生成 sensor_msgs::PointCloud2 消息
    
    
    
## 6.运行
    0. 运行roscore
    a. 运行图漾相机节点                rosrun ty_rgbd_node ty_rgbd_node 
    b. 运行目标检测 以及 目标分析主节点 roslaunch object_analytics_launch analytics_ncnn.launch
    c. 运行可视化节点                  roslaunch object_analytics_visualization rviz.launch
    d. rqt_graph                      节点结构图
   
           
