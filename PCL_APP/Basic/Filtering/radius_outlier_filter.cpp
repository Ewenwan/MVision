/*
球半径滤波器
#include <pcl/filters/radius_outlier_removal.h>
球半径滤波器与统计滤波器相比更加简单粗暴。
以某点为中心　画一个球计算落在该球内的点的数量，当数量大于给定值时，
则保留该点，数量小于给定值则剔除该点。
此算法运行速度快，依序迭代留下的点一定是最密集的，
但是球的半径和球内点的数目都需要人工指定。

RadiusOutlinerRemoval比较适合去除单个的离群点   ConditionalRemoval 比较灵活，可以根据用户设置的条件灵活过滤

*/
#include <iostream>
#include <pcl/point_types.h>
//#include <pcl/filters/passthrough.h>
//#include <pcl/ModelCoefficients.h>	 //模型系数头文件
//#include <pcl/filters/project_inliers.h> //投影滤波类头文件
#include <pcl/io/pcd_io.h>               //点云文件pcd 读写
#include <pcl/filters/radius_outlier_removal.h>// 球半径滤波器
#include <pcl/visualization/cloud_viewer.h>//点云可视化

// 别名
typedef pcl::PointCloud<pcl::PointXYZ>  Cloud;
using namespace std;
int main (int argc, char** argv)
{
  // 定义　点云对象　指针
   Cloud::Ptr cloud_ptr(new Cloud());
   Cloud::Ptr cloud_filtered_ptr(new Cloud());

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

  std::cerr << "Cloud before filtering半径滤波前: " << std::endl;
  for (size_t i = 0; i < cloud_ptr->points.size (); ++i)
    std::cerr << "    " << cloud_ptr->points[i].x << " " 
                        << cloud_ptr->points[i].y << " " 
                        << cloud_ptr->points[i].z << std::endl;

  // 可使用　strcmp获取指定命令行参数 if (strcmp(argv[1], "-r") == 0)
  // 创建滤波器　
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> Radius;
  // 建立滤波器
  Radius.setInputCloud(cloud_ptr);
  Radius.setRadiusSearch(1.2);//半径为　0.8ｍ
  Radius.setMinNeighborsInRadius (2);//半径内最少需要　２个点
  // 执行滤波
  Radius.filter (*cloud_filtered_ptr);

  // 输出滤波后的点云
  std::cerr << "Cloud after filtering半径滤波后: " << std::endl;
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
