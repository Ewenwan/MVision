/*
 投影滤波　输出投影后的点的坐标
 使用参数化模型投影点云
如何将点投影到一个参数化模型上（平面或者球体等），
参数化模型通过一组参数来设定，对于平面来说使用其等式形式。
在PCL中有特定存储常见模型系数的数据结构。

投影滤波类就是输入点云和投影模型，输出为投影到模型上之后的点云。

#include <pcl/ModelCoefficients.h>        //模型系数头文件
#include <pcl/filters/project_inliers.h> 　//投影滤波类头文件

*/
#include <iostream>
#include <pcl/point_types.h>
//#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>	 //模型系数头文件
#include <pcl/filters/project_inliers.h> //投影滤波类头文件
#include <pcl/io/pcd_io.h>               //点云文件pcd 读写
#include <pcl/visualization/cloud_viewer.h>//点云可视化

// 别名
typedef pcl::PointCloud<pcl::PointXYZ>  Cloud;
using namespace std;
int main (int argc, char** argv)
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

  std::cerr << "Cloud before filtering投影滤波前: " << std::endl;
  for (size_t i = 0; i < cloud_ptr->points.size (); ++i)
    std::cerr << "    " << cloud_ptr->points[i].x << " " 
                        << cloud_ptr->points[i].y << " " 
                        << cloud_ptr->points[i].z << std::endl;

  // 填充ModelCoefficients的值,使用ax+by+cz+d=0平面模型，其中 a=b=d=0,c=1 也就是X——Y平面
  //定义模型系数对象，并填充对应的数据　创建投影滤波模型重会设置模型类型　pcl::SACMODEL_PLANE
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());
  coefficients->values.resize (4);
  coefficients->values[0] = coefficients->values[1] = 0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0;

  // 创建投影滤波模型ProjectInliers对象，使用ModelCoefficients作为投影对象的模型参数
  pcl::ProjectInliers<pcl::PointXYZ> proj;	//创建投影滤波对象
  proj.setModelType (pcl::SACMODEL_PLANE);	//设置对象对应的投影模型　　平面模型
  proj.setInputCloud (cloud_ptr);		//设置输入点云
  proj.setModelCoefficients (coefficients);	//设置模型对应的系数
  proj.filter (*cloud_filtered_ptr);		//投影结果存储

  // 输出滤波后的点云
  std::cerr << "Cloud after filtering投影滤波后: " << std::endl;
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
