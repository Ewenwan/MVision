/*
 提取 点云索引
 根据点云索引提取对应的点云
 */

// STL
#include <iostream>

// PCL
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>

int
main (int, char**)
{
  typedef pcl::PointXYZ PointType;
  typedef pcl::PointCloud<PointType> CloudType;
  CloudType::Ptr cloud (new CloudType);
  cloud->is_dense = false;
  // 生成点云 
  PointType p;
  for (unsigned int i = 0; i < 5; ++i)
  {
    p.x = p.y = p.z = static_cast<float> (i);
    cloud->push_back (p);
  }

  std::cout << "Cloud has " << cloud->points.size () << " points." << std::endl;

  pcl::PointIndices indices;// 取得需要的索引
  indices.indices.push_back (0);
  indices.indices.push_back (2);

  pcl::ExtractIndices<PointType> extract_indices;//索引提取器
  extract_indices.setIndices (boost::make_shared<const pcl::PointIndices> (indices));//设置索引
  extract_indices.setInputCloud (cloud);//设置输入点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr output (new pcl::PointCloud<pcl::PointXYZ>);
  extract_indices.filter (*output);//提取对于索引的点云

//　外点　绿色
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_other(new pcl::PointCloud<pcl::PointXYZ>);
// *cloud_other = *cloud - *output;
  // 移去平面局内点，提取剩余点云
  extract_indices.setNegative (true);
  extract_indices.filter (*cloud_other);
  std::cout << "Output has " << output->points.size () << " points." << std::endl;
  return (0);
}
