/*
直通滤波器 PassThrough　　　　　　　　　　　直接指定保留哪个轴上的范围内的点
#include <pcl/filters/passthrough.h>
如果使用线结构光扫描的方式采集点云，必然物体沿z向分布较广，
但x,y向的分布处于有限范围内。
此时可使用直通滤波器，确定点云在x或y方向上的范围，
可较快剪除离群点，达到第一步粗处理的目的。

// 创建点云对象　指针
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
// 原点云获取后进行滤波
pcl::PassThrough<pcl::PointXYZ> pass;// 创建滤波器对象
pass.setInputCloud (cloud);//设置输入点云
pass.setFilterFieldName ("z");//滤波字段名被设置为Z轴方向
pass.setFilterLimits (0.0, 1.0);//可接受的范围为（0.0，1.0） 
//pass.setFilterLimitsNegative (true);//设置保留范围内 还是 过滤掉范围内
pass.filter (*cloud_filtered); //执行滤波，保存过滤结果在cloud_filtered
*/
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>//直通滤波器 PassThrough　
#include <pcl/visualization/cloud_viewer.h>//点云可视化
// 别名
typedef pcl::PointCloud<pcl::PointXYZ>  Cloud;

int
 main (int argc, char** argv)
{
  // 定义　点云对象　指针
   Cloud::Ptr cloud_ptr(new Cloud());
   Cloud::Ptr cloud_filtered_ptr(new Cloud());
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>);

  // 产生点云数据　
  cloud_ptr->width  = 5;
  cloud_ptr->height = 1;
  cloud_ptr->points.resize (cloud_ptr->width * cloud_ptr->height);
  for (size_t i = 0; i < cloud_ptr->points.size (); ++i)
  {
    cloud_ptr->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud_ptr->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud_ptr->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
  }

  std::cerr << "Cloud before filtering滤波前: " << std::endl;
  for (size_t i = 0; i < cloud_ptr->points.size (); ++i)
    std::cerr << "    " << cloud_ptr->points[i].x << " " 
                        << cloud_ptr->points[i].y << " " 
                        << cloud_ptr->points[i].z << std::endl;

  // 创建滤波器对象　Create the filtering object
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud_ptr);//设置输入点云
  pass.setFilterFieldName ("z");// 定义轴
  pass.setFilterLimits (0.0, 1.0);//　范围
 // pass.setKeepOrganized(true); // 保持 有序点云结构===============
  pass.setFilterLimitsNegative (true);//标志为false时保留范围内的点
  pass.filter (*cloud_filtered_ptr);

  // 输出滤波后的点云
  std::cerr << "Cloud after filtering滤波后: " << std::endl;
  for (size_t i = 0; i < cloud_filtered_ptr->points.size (); ++i)
    std::cerr << "    " << cloud_filtered_ptr->points[i].x << " " 
                        << cloud_filtered_ptr->points[i].y << " " 
                        << cloud_filtered_ptr->points[i].z << std::endl;
  // 程序可视化
  pcl::visualization::CloudViewer viewer("pcd　viewer");// 显示窗口的名字
  viewer.showCloud(cloud_filtered_ptr);
  while (!viewer.wasStopped())
    {
        // Do nothing but wait.
    }

  return (0);
}
