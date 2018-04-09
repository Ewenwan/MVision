/*
体素格 （正方体立体空间内 保留一个点（重心点））
下载桌子点云数据
https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd

体素格滤波器VoxelGrid　　在网格内减少点数量保证重心位置不变　PCLPointCloud2()
下采样
#include <pcl/filters/voxel_grid.h>

注意此点云类型为　pcl::PCLPointCloud2　类型  blob　格子类型
#include <pcl/filters/voxel_grid.h>

  // 转换为模板点云 pcl::PointCloud<pcl::PointXYZ>
  pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);

如果使用高分辨率相机等设备对点云进行采集，往往点云会较为密集。
过多的点云数量会对后续分割工作带来困难。
体素格滤波器可以达到向下采样同时不破坏点云本身几何结构的功能。
点云几何结构 不仅是宏观的几何外形，也包括其微观的排列方式，
比如横向相似的尺寸，纵向相同的距离。
随机下采样虽然效率比体素滤波器高，但会破坏点云微观结构.

使用体素化网格方法实现下采样，即减少点的数量 减少点云数据，
并同时保存点云的形状特征，在提高配准，曲面重建，形状识别等算法速度中非常实用，
PCL是实现的VoxelGrid类通过输入的点云数据创建一个三维体素栅格，
容纳后每个体素内用体素中所有点的重心来近似显示体素中其他点，
这样该体素内所有点都用一个重心点最终表示，对于所有体素处理后得到的过滤后的点云，
这种方法比用体素中心（注意中心和重心）逼近的方法更慢，但是对于采样点对应曲面的表示更为准确。

*/
#include <iostream>
#include <pcl/point_types.h>
//#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>//体素格滤波器VoxelGrid
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
  pcl::PCLPointCloud2::Ptr cloud2_ptr(new pcl::PCLPointCloud2());
  pcl::PCLPointCloud2::Ptr cloud2_filtered_ptr(new pcl::PCLPointCloud2());
  Cloud::Ptr cloud_filtered_ptr(new Cloud);
  // 读取点云文件　填充点云对象
  pcl::PCDReader reader;
  reader.read( "../table_scene_lms400.pcd", *cloud2_ptr);
  if(cloud2_ptr==NULL) { cout << "pcd file read err" << endl; return -1;}
  cout << "PointCLoud before filtering 滤波前数量: " << cloud2_ptr->width * cloud2_ptr->height
       << " data points ( " << pcl::getFieldsList (*cloud2_ptr) << "." << endl;

  // 创建滤波器对象　Create the filtering object
  pcl::VoxelGrid<pcl::PCLPointCloud2> vg;
  // pcl::ApproximateVoxelGrid<pcl::PointXYZ> avg;
  vg.setInputCloud (cloud2_ptr);//设置输入点云
  vg.setLeafSize(0.01f, 0.01f, 0.01f);//　体素块大小　１cm
  vg.filter (*cloud2_filtered_ptr);

  // 输出滤波后的点云信息
  cout << "PointCLoud before filtering 滤波后数量: " << cloud2_filtered_ptr->width * cloud2_filtered_ptr->height
       << " data points ( " << pcl::getFieldsList (*cloud2_filtered_ptr) << "." << endl;
  // 写入内存
  pcl::PCDWriter writer;
  writer.write("table_scene_lms400_downsampled.pcd",*cloud2_filtered_ptr,
               Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);

  // 调用系统可视化命令行显示
  //system("pcl_viewer table_scene_lms400_inliers.pcd");

  // 转换为模板点云 pcl::PointCloud<pcl::PointXYZ>
  pcl::fromPCLPointCloud2 (*cloud2_filtered_ptr, *cloud_filtered_ptr);

  // 程序可视化
  pcl::visualization::CloudViewer viewer("pcd　viewer");// 显示窗口的名字
  viewer.showCloud(cloud_filtered_ptr);
  while (!viewer.wasStopped())
    {
        // Do nothing but wait.
    }

  return (0);
}
