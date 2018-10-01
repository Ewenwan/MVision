##  点云分割处理器 segmenter

      segmenter   object_analytics_nodelet::segmenter
                                       2. 点云分割处理器 segmenter
                                         // 订阅 点云话题消息
                                         sub_ = nh.subscribe(Const::kTopicPC2, 1, &SegmenterNodelet::cbSegment, this);
                                         // 发布 点云物体数组
                                         pub_ = nh.advertise<object_analytics_msgs::ObjectsInBoxes3D>(Const::kTopicSegmentation, 1);
                                         ObjectsInBoxes3D ： x，y，z坐标最大最小值，投影到rgb图像平面上的 ROI框
##   点云分割处理器 segmenter
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
        
