/*
平面模型分割
基于随机采样一致性
*/
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>     // 可视化
#include <pcl/filters/extract_indices.h>//按索引提取点云
#include <boost/make_shared.hpp>
int
 main (int argc, char** argv)
{
 // 新建点云 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // 随机生成点云
  cloud->width  = 15;
  cloud->height = 1;//无序点云
  cloud->points.resize (cloud->width * cloud->height);
  // 生成
  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud->points[i].z = 1.0;//都在z＝１平面上
  }

  // 设置一些外点 , 即重新设置几个点的z值，使其偏离z为1的平面
  cloud->points[0].z = 2.0;
  cloud->points[3].z = -2.0;
  cloud->points[6].z = 4.0;
  // 打印点云信息
  std::cerr << "Point cloud data: " << cloud->points.size () << " points" << std::endl;
  for (size_t i = 0; i < cloud->points.size (); ++i)
    std::cerr << "    " << cloud->points[i].x << " "
                        << cloud->points[i].y << " "
                        << cloud->points[i].z << std::endl;
// 模型系数
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);//内点索引
 // pcl::PointIndices::Ptr outliers (new pcl::PointIndices);//外点索引
// 创建一个点云分割对象
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // 是否优化模型系数
  seg.setOptimizeCoefficients (true);
  // 设置模型　和　采样方法
  seg.setModelType (pcl::SACMODEL_PLANE);//　平面模型
  seg.setMethodType (pcl::SAC_RANSAC);// 随机采样一致性算法
  seg.setDistanceThreshold (0.01);//是否在平面上的阈值

  seg.setInputCloud (cloud);//输入点云
  seg.segment (*inliers, *coefficients);//分割　得到平面系数　已经在平面上的点的　索引
 // seg.setNegative(true);//设置提取内点而非外点
  //seg.segment(*inliers, *coefficients);//分割　得到平面系数　已经在平面上的点的　索引

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return (-1);
  }
// 打印平面系数
  std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;
//打印平面上的点
  std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
  for (size_t i = 0; i < inliers->indices.size (); ++i)
    std::cerr << inliers->indices[i] << "    " << cloud->points[inliers->indices[i]].x << " "
                                               << cloud->points[inliers->indices[i]].y << " "
                                               << cloud->points[inliers->indices[i]].z << std::endl;


// 3D点云显示 源点云　绿色
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
  viewer.setBackgroundColor (255, 255, 255);//背景颜色　白色
  //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud, 1.0, 1.0, 0.0);
  //viewer.addPointCloud (cloud, color_handler, "raw point");//添加点云
  //viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "raw point");
  //viewer.addCoordinateSystem (1.0);
  viewer.initCameraParameters ();

//按照索引提取点云　　内点
  pcl::ExtractIndices<pcl::PointXYZ> extract_indices;//索引提取器
  extract_indices.setIndices (boost::make_shared<const pcl::PointIndices> (*inliers));//设置索引
  extract_indices.setInputCloud (cloud);//设置输入点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr output (new pcl::PointCloud<pcl::PointXYZ>);
  extract_indices.filter (*output);//提取对于索引的点云 内点
  std::cerr << "output point size : " << output->points.size () << std::endl;

//平面上的点云　红色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_handler(output, 255, 0, 0);
  viewer.addPointCloud (output, output_handler, "plan point");//添加点云
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "plan point");


//　外点　绿色
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_other(new pcl::PointCloud<pcl::PointXYZ>);
  // *cloud_other = *cloud - *output;
  // 移去平面局内点，提取剩余点云
  extract_indices.setNegative (true);
  extract_indices.filter (*cloud_other);
  std::cerr << "other point size : " << cloud_other->points.size () << std::endl;

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_other_handler(cloud_other, 0, 255, 0);
  viewer.addPointCloud (cloud_other, cloud_other_handler, "other point");//添加点云
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "other point");


    while (!viewer.wasStopped()){
        viewer.spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

  return (0);
}
