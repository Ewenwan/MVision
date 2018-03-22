/*
去除点云里的NAN点
 */

// STL
#include <iostream>
#include <limits>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>

int
main (int, char**)
{
  typedef pcl::PointCloud<pcl::PointXYZ> CloudType;
  CloudType::Ptr cloud (new CloudType);
  cloud->is_dense = false;//非稠密
  CloudType::Ptr output_cloud (new CloudType);
  // 添加NAN点
  CloudType::PointType p_nan;
  p_nan.x = std::numeric_limits<float>::quiet_NaN();
  p_nan.y = std::numeric_limits<float>::quiet_NaN();
  p_nan.z = std::numeric_limits<float>::quiet_NaN();
  cloud->push_back(p_nan);
  // 添加有效点 
  CloudType::PointType p_valid;
  p_valid.x = 1.0f;
  cloud->push_back(p_valid);

  std::cout << "size: " << cloud->points.size () << std::endl;
  for (int i = 0; i < cloud->points.size (); ++i)
       std::cout << cloud->points[i].x << " "
		 << cloud->points[i].y << " "
 		 << cloud->points[i].z << std::endl;
  // 去除Nan点 
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *output_cloud, indices);
  std::cout << "size: " << output_cloud->points.size () << std::endl;
  for (int i = 0; i < output_cloud->points.size (); ++i)
       std::cout << output_cloud->points[i].x << " "
		 << output_cloud->points[i].y << " "
 		 << output_cloud->points[i].z << std::endl;
  return 0;
}
