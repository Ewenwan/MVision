/*
把点云投影到平面上，在平面模型上提取凸（凹）多边形
最大轮廓

本例子先
对点云　直通滤波
使用采样一致性分割算法　提取平面模型，
再通过该估计的平面模型系数从滤波后的点云，投影一组点集　到　投影平面上(投影滤波算法)，
最后为投影后的平面点云　计算其对应的　二维凸多边形（凸包　包围盒）

*/
#include <pcl/ModelCoefficients.h>             //采样一致性模型系数　相关类头文件
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>// 采样一致性
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>	      // 直通滤波器
#include <pcl/filters/project_inliers.h>      //　投影滤波
#include <pcl/segmentation/sac_segmentation.h>//　基于采样一致性分割类定义的头文件
#include <pcl/surface/concave_hull.h>         //　创建凹多边形类定义头文件

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>), //源点云
                                      cloud_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>),//滤波后的点云 
                                      cloud_projected_ptr (new pcl::PointCloud<pcl::PointXYZ>);//投影到平面的点云
  pcl::PCDReader reader;

  reader.read ("../table_scene_mug_stereo_textured.pcd", *cloud_ptr);
//【１】 建立　直通滤波器　消除杂散的NaN (z轴上固定范围内的保留)
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud_ptr);            //设置输入点云
  pass.setFilterFieldName ("z");             //设置分割字段为z坐标
  pass.setFilterLimits (0, 1.1);             //设置分割范围为(0, 1.1)
  pass.filter (*cloud_filtered_ptr);              
  std::cerr << "PointCloud after filtering has: "
            << cloud_filtered_ptr->points.size () << " data points." << std::endl;
// 【2】点云采样一致性分割算法　提取平面模型
  pcl::ModelCoefficients::Ptr coefficients_ptr (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_ptr (new pcl::PointIndices);   //inliers存储分割后的点云
  // 创建分割对象
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // 设置优化系数，该参数为可选参数
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);  // 平面模型
  seg.setMethodType (pcl::SAC_RANSAC);//　采样一致性分割算法
  seg.setDistanceThreshold (0.01);//　内点距离阈值
  seg.setInputCloud (cloud_filtered_ptr);
  seg.segment (*inliers_ptr, *coefficients_ptr);//分割　得出点云　和　平面模型
  std::cerr << "PointCloud after segmentation has: "
            << inliers_ptr->indices.size () << " inliers." << std::endl;

//【3】 通过该估计的平面模型系数从滤波后的点云，投影一组点集　到　投影平面上 
  pcl::ProjectInliers<pcl::PointXYZ> proj; //点云投影滤波模型
  proj.setModelType (pcl::SACMODEL_PLANE); //设置投影模型
  proj.setIndices (inliers_ptr); //索引            
  proj.setInputCloud (cloud_filtered_ptr);//　
  proj.setModelCoefficients (coefficients_ptr); //将估计得到的平面coefficients参数设置为投影平面模型系数
  proj.filter (*cloud_projected_ptr);           //得到投影后的点云
  std::cerr << "PointCloud after projection has: "
            << cloud_projected_ptr->points.size () << " data points." << std::endl;

//【4】提取 平面点云　　的　二维凸多边形（凸包　包围盒）　并　存储　
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConcaveHull<pcl::PointXYZ> chull;        //创建多边形提取对象
  chull.setInputCloud (cloud_projected_ptr);    //设置输入点云为提取后点云
  chull.setAlpha (0.1);//　系数
  chull.reconstruct (*cloud_hull_ptr);          //创建提取创建凹多边形

  std::cerr << "Concave hull has: " << cloud_hull_ptr->points.size ()
            << " data points." << std::endl;

  pcl::PCDWriter writer;
  writer.write ("table_scene_mug_stereo_textured_hull.pcd", *cloud_hull_ptr, false);

  return (0);
}
