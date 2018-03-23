/*
均匀采样：半径求体内 保留一个点（重心点）
下载桌子点云数据
https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd

均匀采样：这个类基本上是相同的，但它输出的点云索引是选择的关键点,是在计算描述子的常见方式。
原理同体素格 （正方体立体空间内 保留一个点（重心点））
而 均匀采样：半径求体内 保留一个点（重心点）

*/
#include <iostream>
#include <pcl/point_types.h>
//#include <pcl/filters/passthrough.h>
//#include <pcl/filters/voxel_grid.h>//体素格滤波器VoxelGrid
#include <pcl/filters/uniform_sampling.h>//均匀采样
#include <pcl/io/pcd_io.h>//点云文件pcd 读写
#include <pcl/visualization/cloud_viewer.h>//点云可视化
#include <pcl_conversions/pcl_conversions.h>//点云类型转换
/*
CloudViewer是简单显示点云的可视化工具，可以使用比较少的代码查看点云，
但是这个是不能用于多线程应用程序当中的。

下面的代码的工作是关于如何在可视化线程中运行代码的例子，
PCLVisualizer是CloudViewer的后端，但它在自己的线程中运行，
如果要使用PCLVisualizer类必须使用调用函数，这样可以避免可视化的并发问题。
但是在实际调用的时候要注意，以防出现核心已转储这一类很麻烦的问题。

*/
using namespace std;
// 别名
typedef pcl::PointCloud<pcl::PointXYZ>  Cloud;

int
 main (int argc, char** argv)
{
  // 定义　点云对象　指针
  //pcl::PCLPointCloud2::Ptr cloud2_ptr(new pcl::PCLPointCloud2());
  //pcl::PCLPointCloud2::Ptr cloud2_filtered_ptr(new pcl::PCLPointCloud2());
  Cloud::Ptr cloud_ptr(new Cloud);
  Cloud::Ptr cloud_filtered_ptr(new Cloud);
  // 读取点云文件　填充点云对象
  pcl::PCDReader reader;
  reader.read( "../table_scene_lms400.pcd", *cloud_ptr);
  if(cloud_ptr==NULL) { cout << "pcd file read err" << endl; return -1;}
  cout << "PointCLoud before filtering 滤波前数量: " << cloud_ptr->width * cloud_ptr->height
       << " data points ( " << pcl::getFieldsList (*cloud_ptr) << "." << endl;

  // 创建滤波器对象　Create the filtering object
    pcl::UniformSampling<pcl::PointXYZ> filter;// 均匀采样
    filter.setInputCloud(cloud_ptr);//输入点云
    filter.setRadiusSearch(0.01f);//设置半径
    //pcl::PointCloud<int> keypointIndices;// 索引
    filter.filter(*cloud_filtered_ptr);
    //pcl::copyPointCloud(*cloud_ptr, keypointIndices.points, *cloud_filtered_ptr);
  // 输出滤波后的点云信息
  cout << "PointCLoud before filtering 滤波后数量: " << cloud_filtered_ptr->width * cloud_filtered_ptr->height
       << " data points ( " << pcl::getFieldsList (*cloud_filtered_ptr) << "." << endl;
  // 写入内存
  //pcl::PCDWriter writer;
  //writer.write("table_scene_lms400_UniformSampled.pcd",*cloud_filtered_ptr,
  //             Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);

  // 调用系统可视化命令行显示
  //system("pcl_viewer table_scene_lms400_inliers.pcd");

  // 转换为模板点云 pcl::PointCloud<pcl::PointXYZ>
 // pcl::fromPCLPointCloud2 (*cloud2_filtered_ptr, *cloud_filtered_ptr);

  // 程序可视化
  pcl::visualization::CloudViewer viewer("pcd　viewer");// 显示窗口的名字
  viewer.showCloud(cloud_filtered_ptr);
  while (!viewer.wasStopped())
    {
        // Do nothing but wait.
    }

  return (0);
}
