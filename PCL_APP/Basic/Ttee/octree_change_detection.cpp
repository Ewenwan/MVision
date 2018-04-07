/*
无序点云数据集的空间变化检测
 
pcl::octree::OctreePointCloudChangeDetector


octree是一种用于管理稀疏3D数据的树状数据结构，
我们学习如何利用octree实现用于多个无序点云之间的空间变化检测，
这些点云可能在尺寸、分辨率、密度和点顺序等方面有所差异。
通过递归地比较octree的树结构，可以鉴定出由octree产生的
体素组成之间的区别所代表的空间变化，
此外，我们解释了如何使用PCL的octree“双缓冲”技术，
以便能实时地探测多个点云之间的空间组成差异。


*/
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_pointcloud_changedetector.h>

#include <iostream>
#include <vector>
#include <ctime>

int
main (int argc, char** argv)
{
  srand ((unsigned int) time (NULL));//用系统时间初始化随机种子

  //=====【1】八叉树分辨率即体素的大小===========
  float resolution = 32.0f;

  //=====【2】初始化空间变化检测对象===============
  pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree (resolution);

  // ======【3】新建一个点云 cloudA  创建点云实例cloudA生成的点云数据用于建立八叉树octree对象
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA_ptr (new pcl::PointCloud<pcl::PointXYZ> );
  pcl::PointCloud<pcl::PointXYZ>& cloudA = *cloudA_ptr;
  // 为点云 cloudA 产生数据  所生成的点数据用于建立八叉树octree对象。
  cloudA.width = 128;
  cloudA.height = 1;//无序点
  cloudA.points.resize (cloudA.width * cloudA.height);//总数
  for (size_t i = 0; i < cloudA.points.size (); ++i)//循环填充
  {
    cloudA.points[i].x = 64.0f * rand () / (RAND_MAX + 1.0f);
    cloudA.points[i].y = 64.0f * rand () / (RAND_MAX + 1.0f);
    cloudA.points[i].z = 64.0f * rand () / (RAND_MAX + 1.0f);
  }

  //======【4】添加点云到八叉树，构建八叉树 ========
  octree.setInputCloud (cloudA_ptr);//设置输入点云
  octree.addPointsFromInputCloud ();//从输入点云构建八叉树

 /***********************************************************************************
    点云cloudA是参考点云用其建立的八叉树对象描述它的空间分布，octreePointCloudChangeDetector
    类继承自Octree2BufBae类，Octree2BufBae类允许同时在内存中保存和管理两个octree，另外它应用了内存池
    该机制能够重新利用已经分配了的节点对象，因此减少了在生成点云八叉树对象时昂贵的内存分配和释放操作
    通过访问 octree.switchBuffers ()重置八叉树 octree对象的缓冲区，但把之前的octree数据仍然保留在内存中
   ************************************************************************************/
  octree.switchBuffers ();// 交换八叉树缓存，但是cloudA对应的八叉树结构仍在内存中

// ======【5】新建另一个点云 cloudB=====================
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB_ptr (new pcl::PointCloud<pcl::PointXYZ> );
  pcl::PointCloud<pcl::PointXYZ>& cloudB = *cloudB_ptr;
  // 为点云 cloudB 产生数据  该点云用于建立新的八叉树结构，
  // 该新的八叉树与前一个cloudA对应的八叉树共享octree对象，但同时在内存中驻留。
  cloudB.width = 128;
  cloudB.height = 1;
  cloudB.points.resize (cloudB.width * cloudB.height);
  for (size_t i = 0; i < cloudB.points.size (); ++i)
  {
    cloudB.points[i].x = 64.0f * rand () / (RAND_MAX + 1.0f);
    cloudB.points[i].y = 64.0f * rand () / (RAND_MAX + 1.0f);
    cloudB.points[i].z = 64.0f * rand () / (RAND_MAX + 1.0f);
  }
  //添加 cloudB到八叉树
  octree.setInputCloud (cloudB_ptr);
  octree.addPointsFromInputCloud ();
/**************************************
为了检索到获取存在于cloudB的点集R，此R并没有cloudA中元素，   B - B交A
可以调用getPointIndicesFromNewVoxels方法，通过探测两个八叉树之间体素的不同，
它返回cloudB中新加点的索引的向量，通过索引向量可以获取R点集。
很明显，这样就探测了cloudB相对于cloudA变化的点集，
但是只能探测在cloudA上增加的点集，而不能探测在cloudA上减少的点集。
**************************************/

  std::vector<int> newPointIdxVector; //存储新加点的索引的向量
  //获取前一cloudA对应的八叉树 在 cloudB对应八叉树中 A内没有的点集
  octree.getPointIndicesFromNewVoxels (newPointIdxVector);

  //打印结果点到标准输出
  std::cout << "Output from getPointIndicesFromNewVoxels:" << std::endl;
  for (size_t i = 0; i < newPointIdxVector.size (); ++i)
    std::cout << i << "# Index:" << newPointIdxVector[i]
              << "  Point: " 
	      << cloudB.points[newPointIdxVector[i]].x << " "
              << cloudB.points[newPointIdxVector[i]].y << " "
              << cloudB.points[newPointIdxVector[i]].z << std::endl;
return 0;
}
