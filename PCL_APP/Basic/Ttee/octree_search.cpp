/*
https://blog.csdn.net/u013019296/article/details/70052311

基于Octree的空间划分及搜索操作
pcl::octree::OctreePointCloudSearch  octree
octree.voxelSearch
octree.nearestKSearch
octree.radiusSearch


octree是一种用于管理稀疏3D数据的树状数据结构，
每个内部节点都正好有八个子节点，本小节中我们学习如何用octree在点云数据中进行空间划分及近邻搜索，
特别地，解释了如何完成
“体素内近邻搜索(Neighbors within Voxel Search)”、
“K近邻搜索(K Nearest Neighbor Search)”和
“半径内近邻搜索(Neighbors within Radius Search)”。

所谓K近邻算法，即是给定一个训练数据集，对新的输入实例，
在训练数据集中找到与该实例最邻近的K个实例（也就是上面所说的K个邻居），
 这K个实例的多数属于某个类，就把该输入实例分类到这个类中。


*/
#include <pcl/point_cloud.h>   //点云头文件
#include <pcl/octree/octree.h>  //八叉树头文件

#include <iostream>
#include <vector>
#include <ctime>

int
main (int argc, char** argv)
{
  //用系统时间初始化随机种子与 srand (time (NULL))的区别
  srand ((unsigned int) time (NULL));//随机数种子

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>& cloud = *cloud_ptr;//引用

  // 产生点云 1000个
  cloud.width = 1000;
  cloud.height = 1;
  cloud.points.resize (cloud.width * cloud.height);

  for (size_t i = 0; i < cloud.points.size (); ++i)//随机循环产生点云的坐标值
  {
    cloud.points[i].x = 1024.0f * rand () / (RAND_MAX + 1.0f);
    cloud.points[i].y = 1024.0f * rand () / (RAND_MAX + 1.0f);
    cloud.points[i].z = 1024.0f * rand () / (RAND_MAX + 1.0f);
  }
/********************************
然后创建一个octree实例，
用设置分辨率进行初始化，
该octree用它的叶节点存放点索引向量，
该分辨率参数描述最低一级octree的最小体素的尺寸，
因此octree的深度是分辨率和点云空间维数的函数，
如果知道点云的边界框，
应该用defineBoundingBox方法把它分配给octree，
然后通过点云指针把所有点增加到octree中。
********************************/
  float resolution = 128.0f;//八叉树分辨率即体素(长宽高)的大小
  //初始化octree
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (resolution);
  octree.setInputCloud (cloud_ptr);//输入点云 指针
  octree.addPointsFromInputCloud ();//构建octree

  pcl::PointXYZ searchPoint;//需要搜索的点
  searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);

 /*************************************************************************************
  一旦PointCloud和octree联系一起，就能进行搜索操作，这里使用的是“体素近邻搜索”，把查询点所在体素中
   其他点的索引作为查询结果返回，结果以点索引向量的形式保存，因此搜索点和搜索结果之间的距离取决于octree的分辨率参数
*****************************************************************************************/
  // ====【1】==体素内近邻搜索 Neighbors within voxel search   方块内
	  std::vector<int> pointIdxVec;//存储体素近邻搜索的结果向量
	  if(octree.voxelSearch(searchPoint,pointIdxVec))    //执行搜索，返回结果到pointIdxVe
	  {
	// 打印搜索点信息
	    std::cout << "Neighbors within voxel search at (" 
	     << searchPoint.x << " " 
	     << searchPoint.y << " " 
	     << searchPoint.z << ")" 
	     << std::endl;
	// 打印寻找到的体素近邻点
	   for (size_t i = 0; i < pointIdxVec.size (); ++i)
	   std::cout << " " << cloud.points[pointIdxVec[i]].x 
	       	     << " " << cloud.points[pointIdxVec[i]].y 
	       	     << " " << cloud.points[pointIdxVec[i]].z << std::endl;
	  }

/**********************************************************************************
  K 被设置为10 ，K近邻搜索  方法把搜索结果写到两个分开的向量，第一个pointIdxNKNSearch包含搜索结果
   （结果点的索引的向量）  第二个向量pointNKNSquaredDistance存储搜索点与近邻之间的距离的平方。
*************************************************************************************/
  // 【2】==== K近邻搜索 ==K nearest neighbor search 

	  int K = 10;//搜寻点 附近的 点数
	  std::vector<int> pointIdxNKNSearch;//存储k近邻搜索 点索引结果
	  std::vector<float> pointNKNSquaredDistance; //与上面对应的平方距离

	  std::cout << "K nearest neighbor search at (" 
		    << searchPoint.x << " " 
		    << searchPoint.y << " " 
		    << searchPoint.z << ") with K=" 
		    << K << std::endl;

	  if (octree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	  {
	    for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
	      std::cout << " " << cloud.points[ pointIdxNKNSearch[i] ].x 
		        << " " << cloud.points[ pointIdxNKNSearch[i] ].y 
		        << " " << cloud.points[ pointIdxNKNSearch[i] ].z 
		        << " (squared distance: "
		        << pointNKNSquaredDistance[i] << ")" << std::endl;
	  }

  //====【3】=====半径内近邻搜索====Neighbors within radius search   球半径内

	  std::vector<int> pointIdxRadiusSearch;//半径内近邻搜索 点索引结果
	  std::vector<float> pointRadiusSquaredDistance;//与上面对应的平方距离

	  float radius = 256.0f * rand () / (RAND_MAX + 1.0f);//随机半价

	  std::cout << "Neighbors within radius search at (" 
		    << searchPoint.x << " " 
		    << searchPoint.y << " " 
		    << searchPoint.z << ") with radius=" 
		    << radius << std::endl;

	  if (octree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	  {
	    for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
	      std::cout << " " << cloud.points[ pointIdxRadiusSearch[i] ].x 
		        << " " << cloud.points[ pointIdxRadiusSearch[i] ].y 
		        << " " << cloud.points[ pointIdxRadiusSearch[i] ].z 
		        << " (squared distance: " 
			<< pointRadiusSquaredDistance[i] << ")" 
			<< std::endl;
	  }
  return 0;
}
