/*基于颜色的区域生长分割法
除了普通点云之外，还有一种特殊的点云，成为RGB点云
。显而易见，这种点云除了结构信息之外，还存在颜色信息。
将物体通过颜色分类，是人类在辨认果实的 过程中进化出的能力，
颜色信息可以很好的将复杂场景中的特殊物体分割出来。
比如Xbox Kinect就可以轻松的捕捉颜色点云。
基于颜色的区域生长分割原理上和基于曲率，法线的分割方法是一致的。
只不过比较目标换成了颜色，去掉了点云规模上 限的限制。
可以认为，同一个颜色且挨得近，是一类的可能性很大，不需要上限来限制。
所以这种方式比较适合用于室内场景分割。
尤其是复杂室内场景，颜色分割 可以轻松的将连续的场景点云变成不同的物体。
哪怕是高低不平的地面，设法用采样一致分割器抽掉平面，
颜色分割算法对不同的颜色的物体实现分割。

算法分为两步：

（1）分割，当前种子点和领域点之间色差小于色差阀值的视为一个聚类
（2）合并，聚类之间的色差小于色差阀值和并为一个聚类，
　　且当前聚类中点的数量小于聚类点数量的与最近的聚类合并在一起
*/

#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>//搜索　kd树
#include <pcl/visualization/cloud_viewer.h>//可视化
#include <pcl/filters/passthrough.h>//直通滤波器
#include <pcl/segmentation/region_growing_rgb.h>//基于颜色的区域增长点云分割算法

int
main (int argc, char** argv)
{
  // 搜索算法
  pcl::search::Search <pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);
  //点云的类型
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZRGB>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZRGB> ("../region_growing_rgb_tutorial.pcd", *cloud) == -1 )
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }
  //存储点云索引　的容器
  pcl::IndicesPtr indices (new std::vector <int>);
  //直通滤波在Z轴的0到1米之间 剔除　nan　和　噪点
  pcl::PassThrough<pcl::PointXYZRGB> pass;// 
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);//直通滤波后的的点云的索引　避免拷贝
  
 //基于颜色的区域生成的对象
  pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
  reg.setInputCloud (cloud);
  reg.setIndices (indices);   //点云的索引
  reg.setSearchMethod (tree);
  reg.setDistanceThreshold (10);//距离的阀值
  reg.setPointColorThreshold (6);//点与点之间颜色容差
  reg.setRegionColorThreshold (5);//区域之间容差
  reg.setMinClusterSize (600);    //设置聚类的大小
  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);//

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer ("Cluster viewer");
  viewer.showCloud (colored_cloud);
  while (!viewer.wasStopped ())
  {
    boost::this_thread::sleep (boost::posix_time::microseconds (100));
  }

  return (0);
}
