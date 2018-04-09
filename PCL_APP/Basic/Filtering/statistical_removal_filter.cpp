/*
统计滤波器用于去除明显离群点
下载桌子点云数据
https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd

统计滤波器 StatisticalOutlierRemoval
#include <pcl/filters/statistical_outlier_removal.h>

统计滤波器用于去除明显离群点（离群点往往由测量噪声引入）。
其特征是在空间中分布稀疏，可以理解为：每个点都表达一定信息量，
某个区域点越密集则可能信息量越大。噪声信息属于无用信息，信息量较小。
所以离群点表达的信息可以忽略不计。考虑到离群点的特征，
则可以定义某处点云小于某个密度，既点云无效。计算每个点到其最近的k(设定)个点平均距离
。则点云中所有点的距离应构成高斯分布。给定均值与方差，可剔除ｎ个西格玛之外的点

激光扫描通常会产生密度不均匀的点云数据集，另外测量中的误差也会产生稀疏的离群点，
此时，估计局部点云特征（例如采样点处法向量或曲率变化率）时运算复杂，
这会导致错误的数值，反过来就会导致点云配准等后期的处理失败。

解决办法：对每个点的邻域进行一个统计分析，并修剪掉一些不符合标准的点。
具体方法为在输入数据中对点到临近点的距离分布的计算，对每一个点，
计算它到所有临近点的平均距离（假设得到的结果是一个高斯分布，
其形状是由均值和标准差决定），那么平均距离在标准范围之外的点，
可以被定义为离群点并从数据中去除。

*/
#include <iostream>
#include <pcl/point_types.h>
//#include <pcl/filters/passthrough.h>
//#include <pcl/filters/voxel_grid.h>//体素格滤波器VoxelGrid
#include <pcl/filters/statistical_outlier_removal.h>//统计滤波器 
#include <pcl/io/pcd_io.h>//点云文件pcd 读写
#include <pcl/visualization/cloud_viewer.h>//点云可视化

using namespace std;
// 别名
typedef pcl::PointCloud<pcl::PointXYZ>  Cloud;

int
 main (int argc, char** argv)
{
  // 定义　点云对象　指针
  Cloud::Ptr cloud_ptr (new Cloud);
  Cloud::Ptr cloud_filtered_ptr (new Cloud);
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>);

  // 读取点云文件　填充点云对象
  pcl::PCDReader reader;
  reader.read( "../table_scene_lms400.pcd", *cloud_ptr);
  if(cloud_ptr==NULL) { cout << "pcd file read err" << endl; return -1;}
  cout << "PointCLoud before filtering 滤波前数量: " << cloud_ptr->width * cloud_ptr->height
       << " data points ( " << pcl::getFieldsList (*cloud_ptr) << "." << endl;

  // 创建滤波器，对每个点分析的临近点的个数设置为50 ，并将标准差的倍数设置为1  这意味着如果一
  // 个点的距离超出了平均距离一个标准差以上，则该点被标记为离群点，并将它移除，存储起来
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sta;//创建滤波器对象
  sta.setInputCloud (cloud_ptr);		    //设置待滤波的点云
  sta.setMeanK (50);	     			    //设置在进行统计时考虑查询点临近点数
  sta.setStddevMulThresh (1.0);	   		    //设置判断是否为离群点的阀值
  sta.filter (*cloud_filtered_ptr); 		    //存储内点

  // 输出滤波后的点云信息
  std::cerr << "Cloud after filtering: " << std::endl;
  std::cerr << *cloud_filtered_ptr << std::endl;

  // 写入内存　　保存内点
  pcl::PCDWriter writer;
  writer.write("table_scene_lms400_inliers.pcd",*cloud_filtered_ptr, false);
  // 保存外点　被滤出的点
  sta.setNegative (true);
  sta.filter (*cloud_filtered_ptr);
  writer.write<pcl::PointXYZ> ("table_scene_lms400_outliers.pcd", *cloud_filtered_ptr, false);

  // 调用系统可视化命令行显示
  //system("pcl_viewer table_scene_lms400_inliers.pcd");

  // 程序可视化
  pcl::visualization::CloudViewer viewer("pcd　viewer");// 显示窗口的名字
  viewer.showCloud(cloud_filtered_ptr);
  while (!viewer.wasStopped())
    {
        // Do nothing but wait.
    }

  return (0);
}
