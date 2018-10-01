# 模型
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
