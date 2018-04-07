/*
类似与二分查找算法思想
Kd树按空间划分生成叶子节点，各个叶子节点里存放点数据，其可以按半径搜索或邻区搜索。
PCL中的Kd tree的基础数据结构使用了FLANN以便可以快速的进行邻区搜索。

kd树(K-dimension tree)是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构。
kd树是是一种二叉树，表示对k维空间的一个划分，构造kd树相当于不断地用垂直于坐标轴的超平面将K维空间切分，
构成一系列的K维超矩形区域。kd树的每个结点对应于一个k维超矩形区域。
利用kd树可以省去对大部分数据点的搜索，从而减少搜索的计算量。


kdtree python实现
http://www.cnblogs.com/21207-iHome/p/6084670.html


也可配合分割算法（欧几里得与区域生长算法）
对空间点云团 快速分类聚合（
欧几里得按照距离 
区域生长算法 利用了法线，曲率，颜色（RGB点云）等信息
）
  // kdtree对象
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  //近邻搜索
  kdtree.nearestKSearch
  //半价搜索
  kdtree.radiusSearch
*/
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <iostream>
#include <vector>
#include <ctime>//time

int
main (int argc, char** argv)
{
  srand (time (NULL));//随机数  用系统时间初始化随机种子

  time_t begin,end;
  begin = clock();  //开始计时
  // 点云对象指针
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>& cloud = *cloud_ptr;

  // 产生假的点云数据
  cloud.width = 400000;//40万数据点
  cloud.height = 1;
  cloud.points.resize (cloud.width * cloud.height);

  for (size_t i = 0; i < cloud.points.size (); ++i)
  {
    cloud.points[i].x = 1024.0f * rand () / (RAND_MAX + 1.0f);
    cloud.points[i].y = 1024.0f * rand () / (RAND_MAX + 1.0f);
    cloud.points[i].z = 1024.0f * rand () / (RAND_MAX + 1.0f);
  }
  // kdtree对象
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  // 输入点云
  kdtree.setInputCloud (cloud_ptr);
  // 随机定义一个 需要搜寻的点  创建一个searchPoint变量作为查询点
  pcl::PointXYZ searchPoint;
  searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);

  // K 个最近点去搜索 nearest neighbor search
  int K = 10;
  // 两个向量来存储搜索到的K近邻，两个向量中，一个存储搜索到查询点近邻的索引，另一个存储对应近邻的距离平方
  std::vector<int> pointIdxNKNSearch(K);//最近临搜索得到的索引
  std::vector<float> pointNKNSquaredDistance(K);//平方距离

  std::cout << "K nearest neighbor search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with K=" << K << std::endl;
  // 开始搜索
 /***********************************************************************************************
   kdtree 近邻搜索
    template<typename PointT> 
    virtual int pcl::KdTree< PointT >::nearestKSearch  ( const PointT &  p_q,  
                                                        int  k,  
                                                        std::vector< int > &  k_indices,  
                                                        std::vector< float > &  k_sqr_distances  
                                                        )  const [pure virtual] 

    Search for k-nearest neighbors for the given query point. 
   纯虚函数，具体实现在其子类KdTreeFLANN中，其用来进行K 领域搜索
    Parameters:
        [in] the given query point 
        [in] k the number of neighbors to search for  
        [out] the resultant indices of the neighboring points
        [out] the resultant squared distances to the neighboring points 为搜索完成后每个邻域点与查询点的欧式距
    Returns:
        number of neighbors found 
    ********************************************************************************************/
  if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
  {
    for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
      std::cout << " " << cloud.points[ pointIdxNKNSearch[i] ].x 
                << " " << cloud.points[ pointIdxNKNSearch[i] ].y 
                << " " << cloud.points[ pointIdxNKNSearch[i] ].z 
                << " (squared distance: " 
	        << pointNKNSquaredDistance[i] 
		<< ")" 
		<< std::endl;
  }

  /**********************************************************************************
   下面的代码展示查找到给定的searchPoint的某一半径（随机产生）内所有近邻，重新定义两个向量
   pointIdxRadiusSearch  pointRadiusSquaredDistance来存储关于近邻的信息
   ********************************************************************************/

  // 半径内最近领搜索 Neighbors within radius search
  std::vector<int> pointIdxRadiusSearch;//存储查询点近邻索引
  std::vector<float> pointRadiusSquaredDistance;//存储近邻点对应距离平方
  float radius = 256.0f * rand () / (RAND_MAX + 1.0f);//随机产生一个半径
  //打印相关信息
  std::cout << "Neighbors within radius search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with radius=" 
	    << radius << std::endl;
  // 开始搜索
  if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
  {
    for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
      //打印所有近邻坐标
      std::cout << " " << cloud.points[ pointIdxRadiusSearch[i] ].x 
                << " " << cloud.points[ pointIdxRadiusSearch[i] ].y 
                << " " << cloud.points[ pointIdxRadiusSearch[i] ].z 
                << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
  }
    //--------------------------------------------------------------------------------------------
    end = clock();  //结束计时
    double Times =  double(end - begin) / CLOCKS_PER_SEC; //将clock()函数的结果转化为以秒为单位的量

    std::cout<<"time: "<<Times<<"s"<<std::endl;

  return 0;
}

