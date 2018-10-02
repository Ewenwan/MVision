# ros_object_analytics　单帧点云(欧氏距离聚类分割) + Yolo_v2/MobileNet_SSD 物体检测
[项目主页](https://github.com/Ewenwan/ros_object_analytics)

      物体分析　Object Analytics (OA) 是一个ros包，
      支持实时物体检测定位和跟踪(realtime object detection, localization and tracking.)
      使用　RGB-D 相机输入,提供物体分析服务，为开发者开发更高级的机器人高级特性应用， 
      例如智能壁障(intelligent collision avoidance),和语义SLAM(semantic SLAM). 

      订阅:
            RGB-D　相机发布的点云消息[sensor_msgs::PointClould2](http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html) 
      发布话题到:
            物体检测　[object detection](https://github.com/intel/object_msgs), 
            物体跟踪　[object tracking](https://github.com/intel/ros_object_analytics/tree/master/object_analytics_msgs), 
            物体3d定位　[object localization](https://github.com/intel/ros_object_analytics/object_analytics_msgs) 

依赖 优秀的算法：

* 基于 图形处理器(GPU) 运行的目标检测　, [ros_opencl_caffe](https://github.com/Ewenwan/ros_opencl_caffe), 
  Yolo v2 model and [OpenCL Caffe](https://github.com/01org/caffe/wiki/clCaffe#yolo2-model-support) framework

* 基于 视觉处理器（VPU） 运行的目标检测 , [ros_intel_movidius_ncs (devel branch)](https://github.com/Ewenwan/ros_intel_movidius_ncs), with MobileNet_SSD model and Caffe framework. 

      (Movidius神经计算棒,首个基于USB模式的深度学习推理工具和独立的人工智能（AI）加速器)
      英特尔的子公司Movidius宣布推出Movidius Myriad X视觉处理器（VPU），
      该处理器是一款低功耗SoC，主要用于基于视觉的设备的深度学习和AI算法加速，比如无人机、智能相机、VR/AR头盔。
      就在不久前的上个月，Movidius还推出了基于Myriad 2芯片的神经计算棒Movidius Neural Compute Stick。


## 编译依赖　compiling dependencies
  ROS packages from [ros-kinetic-desktop-full](http://wiki.ros.org/kinetic/Installation/Ubuntu)
  * roscpp 
  * nodelet
  * std_msgs
  * sensor_msgs
  * geometry_msgs
  * dynamic_reconfigure
  * pcl_conversions
  * cv_bridge
  * libpcl-all
  * libpcl-all-dev
  * ros-kinetic-opencv3

  其他包　来自intel
  * [object_msgs](https://github.com/intel/object_msgs)
  
  * [ros_intel_movidius_ncs](https://github.com/Ewenwan/ros_intel_movidius_ncs)  VPU运行mobileNetSSD
  *  或者
  * [opencl_caffe](https://github.com/intel/ros_opencl_caffe)　　　　　　　　　  GPU运行yolo_v2
  

        NOTE: 跟踪特征依赖 OpenCV (3.3 preferred, >= 3.2 minimum). 
            ROS Kinetic package "ros-kinetic-opencv3" (where OpenCV **3.3.1** is integrated). 
              However, if you're using an old version of ROS Kinetic (where OpenCV **3.2** is integrated), 
              其他来源 [opencv_contrib](https://github.com/opencv/opencv_contrib). 
              需要编译 opencv_contrib (self-built) 和　opencv (ROS Kinetic provided) 查看 "/opt/ros/kinetic/share/opencv3/package.xml"

## 编译和测试　build and test
  * to build　　编译
  ```bash
  cd ${ros_ws} # "ros_ws" is the catkin workspace root directory where this project is placed in
  catkin_make
  ```
  * to test　　测试　
  ```bash
  catkin_make run_tests
  ```

  * to install　安装
  ```bash
  catkin_make install
  ```

## 附加运行依赖　extra running dependencies
   传感器驱动　RGB-D camera
###  Intel RealSense D400
  * [librealsense2 tag v2.9.1](https://github.com/IntelRealSense/librealsense/tree/v2.9.1) and [realsense_ros_camera tag 2.0.2](https://github.com/intel-ros/realsense/tree/2.0.2) if run with Intel RealSense D400
  ```
  roslaunch realsense_ros_camera rs_rgbd.launch
  ```
### Microsoft XBOX 360 / Kinect
  * [openni_launch](http://wiki.ros.org/openni_launch) or [freenect_launch](http://wiki.ros.org/freenect_launch) and their dependencies if run with Microsoft XBOX 360 Kinect
  ```bash
  roslaunch openni_launch openni.launch
  ```
### Astra Camera
  * [ros_astra_camera](https://github.com/orbbec/ros_astra_camera) if run with Astra Camera
  ```bash
  roslaunch astra_launch astra.launch
  ```

## 运行目标分析 OA  command to launch object_analytics

* launch with OpenCL caffe as detection backend yolo_v2　目标检测后端
   ```bash
   roslaunch object_analytics_launch analytics_opencl_caffe.launch
   ```
   
       这里会直接运行  opencl_caffe_launch)/launch/includes/nodelet.launch 
   
* launch with Movidius NCS as detection backend mobileNetSSD 目标检测后端
   ```bash
   roslaunch object_analytics_launch analytics_movidius_ncs.launch
   ```
   
      这里会直接运行 movidius_ncs_launch)/launch/includes/ncs_stream_detection.launch


  Frequently used options
  * **input_points** Specify arg "input_points" for the name of the topic publishing the [sensor_msgs::PointCloud2](http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html) messages by RGB-D camera. 
  Default is "/camera/depth_registered/points" (topic compliant with [ROS OpenNI launch](http://wiki.ros.org/openni_launch))
  
  * **aging_th** 检测次数间隔　Default is 16，因为检测一次之后, 会使用 tracker 来对矩框进行跟踪，检测频率过快，会对检测产生影响
  
  * **probability_th** 跟踪置信度　Specify the probability threshold for tracking object. Default is "0.5".
  ```bash
  roslaunch object_analytics_launch analytics_movidius_ncs.launch aging_th:=30 probability_th:="0.3"
  ```
  
## 节点订阅的　传感器发布的话题
  RGB图像　object_analytics/rgb ([sensor_msgs::Image](http://docs.ros.org/api/sensor_msgs/html/msg/Image.html))

  点云　object_analytics/pointcloud ([sensor_msgs::PointCloud2](http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html))

## 节点发布的处理后得到的信息

  定位信息(3d边框)　object_analytics/localization ([object_analytics_msgs::ObjectsInBoxes3D](https://github.com/intel/ros_object_analytics/tree/master/object_analytics_msgs/msg))

  跟踪信息 object_analytics/tracking ([object_analytics_msgs::TrackedObjects](https://github.com/intel/ros_object_analytics/tree/master/object_analytics_msgs/msg))

  检测信息(2d边框)object_analytics/detection ([object_msgs::ObjectsInBoxes](https://github.com/intel/object_msgs/tree/master/msg))
  
## 消息类型
object_msgs::Object

      string object_name  # 物体名称 object name
      float32 probability # 检测概率 probability of detected object
      
object_msgs::Objects    

      std_msgs/Header header    # 消息时间戳
      Object[] objects_vector   # 物体数组 
      float32 inference_time_ms # 单位: millisecond, 目标检测器运行的时间

object_msgs::ObjectInBox
      
      Object object                     # 目标检测到的物体
      sensor_msgs/RegionOfInterest roi  # 检测框
            uint32 x_offset // 框的左上角点
            uint32 y_offset
            uint32 height   // 框高度
            uint32 width    // 框宽度
            bool do_rectify

object_msgs::ObjectInBoxes

      std_msgs/Header header        # 时间戳
      ObjectInBox[] objects_vector  # 物体边框数组
      float32 inference_time_ms     # 检测的时间
      
object_analytics_nodelet::model::Object2D

      const sensor_msgs::RegionOfInterest roi_;// 物体边框
            uint32 x_offset // 框的左上角点
            uint32 y_offset
            uint32 height   // 框高度
            uint32 width    // 框宽度
            bool do_rectify
      const object_msgs::Object object_;       // 物体名称 + 概率
            string object_name  # 物体名称 object name
            float32 probability # 检测概率 probability of detected object
      
object_analytics_msgs::ObjectInBox3D

      sensor_msgs/RegionOfInterest roi      # region of interest
            uint32 x_offset // 框的左上角点
            uint32 y_offset
            uint32 height   // 框高度
            uint32 width    // 框宽度
            bool do_rectify
      geometry_msgs/Point32 min    # min and max locate the diagonal of a bounding-box of the detected object whose
            float32 x // 3d点
            float32 y
            float32 z
      geometry_msgs/Point32 max    # x, y and z axis parellel to the axises correspondingly in camera coordinates

object_analytics_msgs::ObjectsInBoxes3D

      std_msgs/Header header            # timestamp 时间戳
      ObjectInBox3D[] objects_in_boxes  # ObjectInBox3D 数组

object_analytics_nodelet::model::Object3D

      sensor_msgs::RegionOfInterest roi_;
            uint32 x_offset // 框的左上角点
            uint32 y_offset
            uint32 height   // 框高度
            uint32 width    // 框宽度
            bool do_rectify
      geometry_msgs::Point32 min_;
            float32 x // 3d点
            float32 y
            float32 z
      geometry_msgs::Point32 max_;
            float32 x // 3d点
            float32 y
            float32 z
            
using Relation = std::pair<Object2D, Object3D>; // 2d框 和 3d点云团 配对pair关系

using RelationVector = std::vector<Relation>;   // 配对关系 数组
      
using Object2DVector = std::vector<Object2D>;   // 2d框数组 

using Object3DVector = std::vector<Object3D>;   // 3d点云团数组 


object_analytics_msgs::TrackedObject

      # 物体 id + 2d边框 .
      int32 id                            # object identifier
      sensor_msgs/RegionOfInterest roi    # region of interest
      
object_analytics_msgs::TrackedObjects

      std_msgs/Header header              # timestamp  时间戳
      TrackedObject[] tracked_objects     # TrackedObject 目标追踪数组 
## PCL定义新点类型 
```c
// PCL新定义点类型   3d点坐标 + 2d的像素坐标值 3d-2d点对
struct PointXYZPixel
{
  PCL_ADD_POINT4D;// 
  uint32_t pixel_x;// 像素值
  uint32_t pixel_y;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;  // NOLINT
// 注册点类型
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZPixel,                // xyz + pixel x, y as fields
                                  (float, x, x)                 // field x
                                  (float, y, y)                 // field y
                                  (float, z, z)                 // field z
                                  (uint32_t, pixel_x, pixel_x)  // field pixel x
                                  (uint32_t, pixel_y, pixel_y)  // field pixel y
                                  )
```

## 点云 索引 得到其对应的像素坐标
```c
// XYZRGBA 点+颜色 点云  拷贝到 XYZ+像素点坐标 点云
void ObjectUtils::copyPointCloud(const PointCloudT::ConstPtr& original, const std::vector<int>& indices,
                                 pcl::PointCloud<PointXYZPixel>::Ptr& dest)
{
  pcl::copyPointCloud(*original, indices, *dest);// 拷贝 3d点坐标
  uint32_t width = original->width;              // 相当于图像宽度
  for (uint32_t i = 0; i < indices.size(); i++)
  {
    dest->points[i].pixel_x = indices[i] % width;// 列坐标
    dest->points[i].pixel_y = indices[i] / width;// 行坐标
  }
}
```
## 获取最大最小值 std::minmax_element()
```c
  auto cmp_x = [](PointXYZPixel const& l, PointXYZPixel const& r) { return l.x < r.x; };// 按x值域大小 的 比较函数
  auto minmax_x = std::minmax_element(point_cloud->begin(), point_cloud->end(), cmp_x);//std库 获取最大最小值
  x_min = *(minmax_x.first);       // 最小值
  x_max = *(minmax_x.second);// 最大值
```

## object_analytics 节点分析
      1. RGBD传感器预处理分割器 splitter  
         输入: /camera/depth_registered/points
         输出: pointcloud   3d点 
         输出: rgb 2d图像
         object_analytics_nodelet/splitter/SplitterNodelet 
         
      2. 点云分割处理器 segmenter
         object_analytics_launch/launch/includes/segmenter.launch
         输入: pointcloud   3d点
         输出: segmentation 分割
         object_analytics_nodelet/segmenter/SegmenterNodelet 
         object_analytics_nodelet/src/segmenter/segmenter_nodelet.cpp
         订阅发布话题后
         std::unique_ptr<Segmenter> impl_;
         点云话题回调函数:
            boost::shared_ptr<ObjectsInBoxes3D> msg = boost::make_shared<ObjectsInBoxes3D>();// 3d框
            msg->header = points->header;
            impl_->segment(points, msg);//检测
            pub_.publish(msg);          //发布检测消息
            
         object_analytics_nodelet/src/segmenter/segmenter.cpp
         a. 首先　ros点云消息转化成 pcl点云消息
               const sensor_msgs::PointCloud2::ConstPtr& points；
               PointCloudT::Ptr pointcloud(new PointCloudT);
               fromROSMsg<PointT>(*points, pcl_cloud);
               
         b. 执行分割　Segmenter::doSegment()
            std::vector<PointIndices> cluster_indices;// 点云所属下标
            PointCloudT::Ptr cloud_segment(new PointCloudT);// 分割点云
              std::unique_ptr<AlgorithmProvider> provider_;
            std::shared_ptr<Algorithm> seg = provider_->get();//　分割算法
            seg->segment(cloud, cloud_segment, cluster_indices);// 执行分割
            
             AlgorithmProvider -> virtual std::shared_ptr<Algorithm> get() = 0;
             Algorithm::segment()
             object_analytics_nodelet/src/segmenter/organized_multi_plane_segmenter.cpp
             class OrganizedMultiPlaneSegmenter : public Algorithm
             OrganizedMultiPlaneSegmenter 类集成　Algorithm类
             分割算法 segment(){} 基于pcl算法
               1. 提取点云法线 OrganizedMultiPlaneSegmenter::estimateNormal()
               2. 分割平面     OrganizedMultiPlaneSegmenter::segmentPlanes()           平面系数模型分割平面
               3. 去除平面后 分割物体  OrganizedMultiPlaneSegmenter::segmentObjects() 　欧氏距离聚类分割
             
         c. 生成消息  Segmenter::composeResult()
            for (auto& obj : objects)
            {
            object_analytics_msgs::ObjectInBox3D oib3;
            oib3.min = obj.getMin();
            oib3.max = obj.getMax();
            oib3.roi = obj.getRoi();
            msg->objects_in_boxes.push_back(oib3);
            }
            
      3. 3d定位器　merger
         输入: 2d检测分割结果 detection
         输入: 3d检测分割结果 segmentation
         输出: 3d定位结果　　 localization
        object_analytics_nodelet/merger/MergerNodelet
        
      4. 目标跟踪器 tracker
         输入: 2d图像        rgb        input_rgb 
         输入: 2d检测分割结果 detection  input_2d 
         输出: 跟踪结果　　　 tracking　　output  
         参数: 跟踪次数阈值:  aging_th：default="30"； // 被跟踪次数长度，年龄？？？，跟踪的次数越多，可能就越不准确===
         参数: 跟踪置信度:    probability_th" default="0.5"
         object_analytics_nodelet/tracker/TrackingNodelet
         
         检测结果的每一个roi会用来初始化一个跟踪器。之后会跟踪后面的每一帧，直到下一个检测结果来到。
         [detection, tracking, tracking, tracking, ..., tracking] [detection, tracking, tracking, tracking, ..., tracking]
         
## object_analytics_visualization 可视化
      5. 3d定位可视化　visualization3d　localization
         输入: 2d检测结果 "detection_topic" default="/object_analytics/detection" 
         输入: 3d检测结果 "localization_topic" default="/object_analytics/localization" 
         输入: 2d跟踪结果 "tracking_topic" default="/object_analytics/tracking" 
         输出: rviz可视化话题　"output_topic" default="localization_rviz" 

      6. 2d跟踪可视化　visualization2d 
         输入: 2d检测结果 "detection_topic" default="/object_analytics/detection" 
         输入: 2d跟踪结果 "tracking_topic" default="/object_analytics/tracking" 
         输入: 2d图像     "image_topic" default="/object_analytics/rgb" 
         输出: rviz可视化话题　""output_topic" default="tracking_rviz"    

## 目标检测接口　
### GPU  yolo_v2　目标检测后端
      opencl_caffe_launch/launch/includes/nodelet.launch 
      
[来源　ros_opencl_caffe　](https://github.com/Ewenwan/ros_opencl_caffe)

      输入: 2d图像          /usb_cam/image_raw    input_topic
      输出: 2d检测分割结果  input_detection        output_topic
      参数文件: param_file default= find opencl_caffe_launch/launch/includes/default.yaml"
            模型文件 net_config_path:  "/opt/clCaffe/models/yolo/yolo416/yolo_fused_deploy.prototxt"
            权重文件 weights_path:     "/opt/clCaffe/models/yolo/yolo416/fused_yolo.caffemodel"
            类别标签文件　labels_path: "/opt/clCaffe/data/yolo/voc.txt"

      节点：　opencl_caffe/opencl_caffe_nodelet
      opencl_caffe/src/nodelet.cpp
      Nodelet::onInit()  --->  loadResources() 
      检测器　detector_.reset(new DetectorGpu());
      载入配置文件　detector_->loadResources(net_config_path, weights_path, labels_path)
      订阅话题回调函数　 sub_ = getNodeHandle().subscribe("/usb_cam/image_raw", 1, &Nodelet::cbImage, this);
      Nodelet::cbImage();
         网络前向推理　detector_->runInference(image_msg, objects)
         发布话题　　　pub_.publish(objects);


      DetectorGpu 类
      opencl_caffe/src/detector_gpu.cpp
      网络初始化:
      net.reset(new caffe::Net<Dtype>(net_cfg, caffe::TEST, caffe::Caffe::GetDefaultDevice()));
      net->CopyTrainedLayersFrom(weights);

      模式：
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
      caffe::Caffe::SetDevice(0);
      载入图像:
      cv::cvtColor(cv_bridge::toCvShare(image_msg, "rgb8")->image, image, cv::COLOR_RGB2BGR);
      initInputBlob(resizeImage(image), input_channels);
      网络前传:
      net->Forward();
      获取网络结果:
      caffe::Blob<Dtype>* result_blob = net->output_blobs()[0];
      const Dtype* result = result_blob->cpu_data();
      const int num_det = result_blob->height();
      检测结果:
      object_msgs::ObjectInBox object_in_box;
      object_in_box.object.object_name = labels_list[classid];
      object_in_box.object.probability = confidence;
      object_in_box.roi.x_offset = left;
      object_in_box.roi.y_offset = top;
      object_in_box.roi.height = bot - top;
      object_in_box.roi.width = right - left;
      objects.objects_vector.push_back(object_in_box);

### VPU   mobileNetSSD 目标检测后端
      movidius_ncs_launch/launch/includes/ncs_stream_detection.launch

[来源　ros_intel_movidius_ncs　](https://github.com/Ewenwan/ros_intel_movidius_ncs)

      输入: 2d图像         input_rgb        input_topic
      输出: 2d检测分割结果  input_detection  output_topic
      参数: 
         模型类型 name="cnn_type" default="mobilenetssd"
         模型配置文件　name="param_file" default="$(find movidius_ncs_launch)/config/mobilenetssd.yaml"
            网络图配置文件 graph_file_path: "/opt/movidius/ncappzoo/caffe/SSD_MobileNet/graph"
            类别文件voc21  category_file_path: "/opt/movidius/ncappzoo/data/ilsvrc12/voc21.txt"
            网络输入尺寸   network_dimension: 300
            通道均值 :
              channel1_mean: 127.5
              channel2_mean: 127.5
              channel3_mean: 127.5
            归一化:
              scale: 0.007843
          节点文件　movidius_ncs_stream/NCSNodelet　movidius_ncs_stream/src/ncs_nodelet.cpp
      输入话题　input_topic ： 2d图像　      /camera/rgb/image_raw
       输出话题  output_topic : 2d检测框结果  detected_objects
          ncs_manager_handle_ = std::make_shared<movidius_ncs_lib::NCSManager>();
          movidius_ncs_lib::NCSManager 来自 movidius_ncs_lib/src/ncs_manager.cpp
          NCSImpl::init(){ }
          订阅 rgb图像的回调函数 cbDetect()
          sub_ = it->subscribe("/camera/rgb/image_raw", 1, &NCSImpl::cbDetect, this);
          cbDetect(){ }
          1. 从话题复制图像
            cv::Mat camera_data = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
          2. 提取检测结果回调函数
            FUNP_D detection_result_callback = boost::bind(&NCSImpl::cbGetDetectionResult, this, _1, _2);
          3. 进行目标检测 
            ncs_manager_handle_->detectStream(camera_data, detection_result_callback, image_msg);
          NCSManager::detectStream(){}
          得到检测结果 : movidius_ncs_lib::DetectionResultPtr result;
          检测结果:
           for (auto item : result->items_in_boxes)
            object_msgs::ObjectInBox obj;
            obj.object.object_name = item.item.category;
            obj.object.probability = item.item.probability;
            obj.roi.x_offset = item.bbox.x;
            obj.roi.y_offset = item.bbox.y;
            obj.roi.width = item.bbox.width;
            obj.roi.height = item.bbox.height;
            objs_in_boxes.objects_vector.push_back(obj);
         发布检测结果:  
            objs_in_boxes.header = header;
            objs_in_boxes.inference_time_ms = result->time_taken;
            pub_.publish(objs_in_boxes);
                  
## KPI of differnt detection backends
<table>
    <tr>
        <td></td>
        <td>topic</td>
        <td>fps</td>
        <td>latency <sup>sec</sup></td>
    </tr>
    <tr>
        <td rowspan='4'>OpenCL Caffe</td>
    </tr>
    <tr>
        <td>localization</td>
        <td>6.63</td>
        <td>0.23</td>
    </tr>
    <tr>
        <td>detection</td>
        <td>8.88</td>
        <td>0.17</td>
    </tr>
    <tr>
        <td>tracking</td>
        <td>12.15</td>
        <td>0.33</td>
    </tr>
    <tr>
        <td rowspan='4'>Movidius NCS</sup></td>
    </tr>
    <tr>
        <td>localization</td>
        <td>7.44</td>
        <td>0.21</td>
    </tr>
    <tr>
        <td>detection</td>
        <td>10.5</td>
        <td>0.15</td>
    </tr>
    <tr>
        <td>tracking</td>
        <td>13.85</td>
        <td>0.24</td>
    </tr>
</table>

* CNN model of Movidius NCS is MobileNet
* Hardware: Intel(R) Xeon(R) CPU E3-1275 v5 @3.60GHz, 32GB RAM, Intel(R) RealSense R45

## rviz中可视化　visualize tracking and localization results on RViz
  Steps to enable visualization on RViz are as following
  ```bash
  roslaunch object_analytics_visualization rviz.launch
  ```
###### *ROS 2 Object Analytics: https://github.com/intel/ros2_object_analytics*
###### *Any security issue should be reported using process at https://01.org/security*


## 程序
```
├── object_analytics           ========================工程汇总===============================
│   ├── CMakeLists.txt
│   └── package.xml
├── object_analytics_launch    ========================工程启动脚本===========================
│   ├── CMakeLists.txt
│   ├── launch
│   │   ├── analytics_movidius_ncs.launch   ==== 基于 NPU + mobilenet-ssd 的目标检测 启动脚本
│   │   ├── analytics_opencl_caffe.launch   ==== 基于 GPU + YOLOV2 的目标检测 启动脚本
│   │   └── includes
│   │       ├── manager.launch              ==== 管理节点
│   │       ├── merger.launch               ==== 2d分割 和 点云分割融合 3d定位器
│   │       ├── nodelet.launch.xml          ==== 项目参数
│   │       ├── segmenter.launch            ==== 点云分割处理器
│   │       ├── splitter.launch             ==== RGBD传感器预处理分割器
│   │       └── tracker.launch              ==== 目标跟踪器
│   └── package.xml
├── object_analytics_msgs     =========================自定义消息msg==========================
│   ├── CMakeLists.txt
│   ├── msg
│   │   ├── ObjectInBox3D.msg               ==== 单个物体3d边框定位信息
│   │   ├── ObjectsInBoxes3D.msg            ==== 多个物体3d边框定位信息 
│   │   ├── TrackedObject.msg               ==== 单个物体跟踪信息
│   │   └── TrackedObjects.msg              ==== 多个物体根系信息
│   └── package.xml
├── object_analytics_nodelet  ======================== 工程源文件节点功能实现==================
│   ├── cfg                                      ==== 动态参数配置========
│   │   ├── OrganizedMultiPlaneSegmentation.cfg  ==== 点云平面分割参数配置
│   │   └── SegmentationAlgorithms.cfg           ==== 点云物体分割算法 欧式距离聚类分割 区域聚类分割算法
│   ├── CMakeLists.txt
│   ├── include                                  ==== 头文件 ========
│   │    └── object_analytics_nodelet
│   │       ├── const.h
│   │       ├── merger                   ==== 2d分割 和 点云分割融合 3d定位器
│   │       │   ├── merger.h
│   │       │   └── merger_nodelet.h
│   │       ├── model                    ==== 模型
│   │       │   ├── object2d.h
│   │       │   ├── object3d.h
│   │       │   └── object_utils.h
│   │       ├── segmenter                ==== 3d点云分割算法
│   │       │   ├── algorithm_config.h
│   │       │   ├── algorithm.h
│   │       │   ├── algorithm_provider.h
│   │       │   ├── algorithm_provider_impl.h
│   │       │   ├── organized_multi_plane_segmenter.h  平面分割
│   │       │   ├── segmenter.h
│   │       │   └── segmenter_nodelet.h
│   │       ├── splitter                ==== RGBD传感器预处理分割器
│   │       │   ├── splitter.h
│   │       │   └── splitter_nodelet.h
│   │       └── tracker                 ==== 目标跟踪器
│   │           ├── tracking.h
│   │           ├── tracking_manager.h
│   │           └── tracking_nodelet.h
│   ├── mainpage.dox
│   ├── object_analytics_nodelet_plugins.xml
│   ├── package.xml
│   ├── src
│   │   ├── const.cpp
│   │   ├── merger                   ==== 2d分割 和 点云分割融合 3d定位器
│   │   │   ├── merger.cpp
│   │   │   └── merger_nodelet.cpp
│   │   ├── model                    ==== 模型
│   │   │   ├── object2d.cpp
│   │   │   ├── object3d.cpp
│   │   │   └── object_utils.cpp
│   │   ├── segmenter                ==== 3d点云分割算法
│   │   │   ├── algorithm_provider_impl.cpp
│   │   │   ├── organized_multi_plane_segmenter.cpp  =平面分割=
│   │   │   ├── segmenter.cpp
│   │   │   └── segmenter_nodelet.cpp
│   │   ├── splitter                  ==== RGBD传感器预处理分割器
│   │   │   ├── splitter.cpp
│   │   │   └── splitter_nodelet.cpp
│   │   └── tracker                   ==== 目标跟踪器
│   │       ├── tracking.cpp
│   │       ├── tracking_manager.cpp
│   │       └── tracking_nodelet.cpp
│   └── tests                        ======= 单元侧测试 =====
│       ├── mock_segmenter_detector.cpp
│       ├── mtest_tracking.cpp
│       ├── mtest_tracking.test
│       ├── nodetest_merger.test
│       ├── nodetest_segmenter.test
│       ├── nodetest_splitter.test
│       ├── pc2_publisher.cpp
│       ├── resource                 ====== 部分点云数据
│       │   ├── copy.pcd
│       │   ├── cup.pcd
│       │   ├── object3d.pcd
│       │   ├── project.pcd
│       │   ├── segment.pcd
│       │   └── split.pcd
│       ├── unittest_merger.cpp
│       ├── unittest_object2d.cpp
│       ├── unittest_object3d.cpp
│       ├── unittest_objectutils.cpp
│       ├── unittest_segmenter.cpp
│       ├── unittest_splitter.cpp
│       ├── unittest_util.cpp
│       └── unittest_util.h.in
├── object_analytics_visualization   ===================  rviz可视化 ===========================
│   ├── cfg
│   │   └── object_analytics_visualization.rviz
│   ├── CMakeLists.txt
│   ├── launch
│   │   ├── includes
│   │   │   ├── localization.launch    ==== 定位可视化
│   │   │   └── tracking.launch        ==== 跟踪可视化
│   │   ├── rviz.launch
│   │   └── viz_all.launch
│   ├── mainpage.dox
│   ├── package.xml
│   └── scripts
│       ├── image_publisher.py         ==== 发布图像
│       └── marker_publisher.py        ==== 发布虚拟物体


```

