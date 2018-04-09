/*
条件滤波器
    可以一次删除满足对输入的点云设定的一个或多个条件指标的所有的数据点
    删除点云中不符合用户指定的一个或者多个条件的数据点
不在条件范围内的点　被替换为　nan
pcl::removeNaNFromPointCloud()去除Nan点
#include <pcl/filters/conditional_removal.h>
*/
#include <iostream>
#include <pcl/point_types.h>
//#include <pcl/filters/passthrough.h>
//#include <pcl/ModelCoefficients.h>	   //模型系数头文件
//#include <pcl/filters/project_inliers.h> //投影滤波类头文件
#include <pcl/io/pcd_io.h>                 //点云文件pcd 读写
//#include <pcl/filters/radius_outlier_removal.h>// 球半径滤波器
#include <pcl/filters/conditional_removal.h>     //条件滤波器

//#include <pcl/filters/filter.h>

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
  //创建条件定义对象
  pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond_cptr(new pcl::ConditionAnd<pcl::PointXYZ>());
  //为条件定义对象添加比较算子
  range_cond_cptr->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
                                  pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, 0.0)));
  //添加在Z字段上大于（pcl::ComparisonOps::GT　great Then）0的比较算子

  range_cond_cptr->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
                                  pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, 0.8)));
  //添加在Z字段上小于（pcl::ComparisonOps::LT　Lower Then）0.8的比较算子

  // 创建滤波器并用条件定义对象初始化
  pcl::ConditionalRemoval<pcl::PointXYZ> conditrem;//创建条件滤波器
  conditrem.setCondition (range_cond_cptr);        //并用条件定义对象初始化            
  conditrem.setInputCloud (cloud_ptr);             //输入点云
  conditrem.setKeepOrganized(true);                //设置保持点云的结构
  // 执行滤波
  conditrem.filter(*cloud_filtered_ptr);           //大于0.0小于0.8这两个条件用于建立滤波器
  // 不在条件范围内的点　被替换为　nan

  // 输出滤波后的点云
  std::cerr << "Cloud after filtering半径滤波后: " << std::endl;
  for (size_t i = 0; i < cloud_filtered_ptr->points.size (); ++i)
    std::cerr << "    " << cloud_filtered_ptr->points[i].x << " " 
                        << cloud_filtered_ptr->points[i].y << " " 
                        << cloud_filtered_ptr->points[i].z << std::endl;
  // 去除　nan点
  std::vector<int> mapping;
  pcl::removeNaNFromPointCloud(*cloud_filtered_ptr, *cloud_filtered_ptr, mapping);
  // 输出去除nan后的点云
  std::cerr << "Cloud after delet nan point去除nan点 : " << std::endl;
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
